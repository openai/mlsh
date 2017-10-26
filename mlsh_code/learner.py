import numpy as np
import tensorflow as tf
from rl_algs.common import explained_variance, fmt_row, zipsame
from rl_algs import logger
import rl_algs.common.tf_util as U
import time
from rl_algs.common.mpi_adam import MpiAdam
from mpi4py import MPI
from collections import deque
from dataset import Dataset

class Learner:
    def __init__(self, env, policy, old_policy, sub_policies, old_sub_policies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64):
        self.policy = policy
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.num_subpolicies = len(sub_policies)
        self.sub_policies = sub_policies
        ob_space = env.observation_space
        ac_space = env.action_space

        # for training theta
        # inputs for training theta
        ob = U.get_placeholder_cached(name="ob")
        ac = policy.pdtype.sample_placeholder([None])
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        total_loss = self.policy_loss(policy, old_policy, ob, ac, atarg, ret, clip_param)
        self.master_policy_var_list = policy.get_trainable_variables()
        self.master_loss = U.function([ob, ac, atarg, ret], U.flatgrad(total_loss, self.master_policy_var_list))
        self.master_adam = MpiAdam(self.master_policy_var_list, comm=comm)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(old_policy.get_variables(), policy.get_variables())])

        self.assign_subs = []
        self.change_subs = []
        self.adams = []
        self.losses = []
        self.sp_ac = sub_policies[0].pdtype.sample_placeholder([None])
        for i in range(self.num_subpolicies):
            varlist = sub_policies[i].get_trainable_variables()
            self.adams.append(MpiAdam(varlist))
            # loss for test
            loss = self.policy_loss(sub_policies[i], old_sub_policies[i], ob, self.sp_ac, atarg, ret, clip_param)
            self.losses.append(U.function([ob, self.sp_ac, atarg, ret], U.flatgrad(loss, varlist)))

            self.assign_subs.append(U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(old_sub_policies[i].get_variables(), sub_policies[i].get_variables())]))
            self.zerograd = U.function([], self.nograd(varlist))

        U.initialize()

        self.master_adam.sync()
        for i in range(self.num_subpolicies):
            self.adams[i].sync()

    def nograd(self, var_list):
        return tf.concat(axis=0, values=[
            tf.reshape(tf.zeros_like(v), [U.numel(v)])
            for v in var_list
        ])


    def policy_loss(self, pi, oldpi, ob, ac, atarg, ret, clip_param):
        ratio = tf.exp(pi.pd.logp(ac) - tf.clip_by_value(oldpi.pd.logp(ac), -20, 20)) # advantage * pnew / pold
        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        pol_surr = - U.mean(tf.minimum(surr1, surr2))
        vfloss1 = tf.square(pi.vpred - ret)
        vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2))
        total_loss = pol_surr + vf_loss
        return total_loss

    def syncMasterPolicies(self):
        self.master_adam.sync()

    def syncSubpolicies(self):
        for i in range(self.num_subpolicies):
            self.adams[i].sync()

    def updateMasterPolicy(self, seg):
        ob, ac, atarg, tdlamret = seg["macro_ob"], seg["macro_ac"], seg["macro_adv"], seg["macro_tdlamret"]
        # ob = np.ones_like(ob)
        mean = atarg.mean()
        std = atarg.std()
        meanlist = MPI.COMM_WORLD.allgather(mean)
        global_mean = np.mean(meanlist)

        real_var = std**2 + (mean - global_mean)**2
        variance_list = MPI.COMM_WORLD.allgather(real_var)
        global_std = np.sqrt(np.mean(variance_list))

        atarg = (atarg - global_mean) / max(global_std, 0.000001)

        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
        optim_batchsize = min(self.optim_batchsize,ob.shape[0])

        self.policy.ob_rms.update(ob) # update running mean/std for policy

        self.assign_old_eq_new()
        for _ in range(self.optim_epochs):
            for batch in d.iterate_once(optim_batchsize):
                g = self.master_loss(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"])
                self.master_adam.update(g, 0.01, 1)

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        logger.record_tabular("EpRewMean", np.mean(rews))

        return np.mean(rews), np.mean(seg["ep_rets"])

    def updateSubPolicies(self, test_segs, num_batches, optimize=True):
        for i in range(self.num_subpolicies):
            is_optimizing = True
            test_seg = test_segs[i]
            ob, ac, atarg, tdlamret = test_seg["ob"], test_seg["ac"], test_seg["adv"], test_seg["tdlamret"]
            if np.shape(ob)[0] < 1:
                is_optimizing = False
            else:
                atarg = (atarg - atarg.mean()) / max(atarg.std(), 0.000001)
            test_d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
            test_batchsize = int(ob.shape[0] / num_batches)

            self.assign_subs[i]() # set old parameter values to new parameter values
            # Here we do a bunch of optimization epochs over the data
            if self.optim_batchsize > 0 and is_optimizing and optimize:
                self.sub_policies[i].ob_rms.update(ob)
                for k in range(self.optim_epochs):
                    m = 0
                    for test_batch in test_d.iterate_times(test_batchsize, num_batches):
                        test_g = self.losses[i](test_batch["ob"], test_batch["ac"], test_batch["atarg"], test_batch["vtarg"])
                        self.adams[i].update(test_g, self.optim_stepsize, 1)
                        m += 1
            else:
                self.sub_policies[i].ob_rms.noupdate()
                blank = self.zerograd()
                for _ in range(self.optim_epochs):
                    for _ in range(num_batches):
                        self.adams[i].update(blank, self.optim_stepsize, 0)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
