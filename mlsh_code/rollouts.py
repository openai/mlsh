import numpy as np
import math
import time

def traj_segment_generator(pi, sub_policies, env, macrolen, horizon, stochastic, args):
    replay = args.replay
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()
    cur_subpolicy = 0
    macro_vpred = 0
    macro_horizon = math.ceil(horizon/macrolen)

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    macro_acs = np.zeros(macro_horizon, 'int32')
    macro_vpreds = np.zeros(macro_horizon, 'float32')

    ob = env.reset()

    x = 0
    z = 0

    # total = [0,0]
    # tt = 0

    while True:
        if t % macrolen == 0:
            cur_subpolicy, macro_vpred = pi.act(stochastic, ob)

            if np.random.uniform() < 0.1:
                cur_subpolicy = np.random.randint(0, len(sub_policies))
            if args.force_subpolicy is not None:
                cur_subpolicy = args.force_subpolicy
                z += 1

        ac, vpred = sub_policies[cur_subpolicy].act(stochastic, ob)
        # if np.random.uniform(0,1) < 0.05:
            # ac = env.action_space.sample()

        if t > 0 and t % horizon == 0:
            # tt += 1
            # print(total)
            # total = [0,0]
            dicti = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news, "ac" : acs, "ep_rets" : ep_rets, "ep_lens" : ep_lens, "macro_ac" : macro_acs, "macro_vpred" : macro_vpreds}
            yield {key: np.copy(val) for key,val in dicti.items()}
            ep_rets = []
            ep_lens = []
            x += 1

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        if t % macrolen == 0:
            macro_acs[int(i/macrolen)] = cur_subpolicy
            macro_vpreds[int(i/macrolen)] = macro_vpred

        ob, rew, new, info = env.step(ac)
        rews[i] = rew

        if replay:
            if len(ep_rets) == 0:
                # if x % 5 == 0:
                env.render()
                    # print(info)
            pass

        cur_ep_ret += rew
        cur_ep_len += 1
        if new and ((t+1) % macrolen == 0):
        # if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_advantage_macro(seg, macrolen, gamma, lam):
    new = np.append(seg["new"][0::macrolen], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["macro_vpred"], 0)
    T = int(len(seg["rew"])/macrolen)
    seg["macro_adv"] = gaelam = np.empty(T, 'float32')
    rew = np.sum(seg["rew"].reshape(-1, macrolen), axis=1)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["macro_tdlamret"] = seg["macro_adv"] + seg["macro_vpred"]

    # print(seg["macro_ac"])
    # print(rew)
    # print(seg["macro_adv"])
    seg["macro_ob"] = seg["ob"][0::macrolen]

def prepare_allrolls(allrolls, macrolen, gamma, lam, num_subpolicies):
    for i in range(len(allrolls) - 1):
        for key,value in allrolls[i + 1].items():
            allrolls[0][key] = np.append(allrolls[0][key], value, axis=0)
    test_seg = allrolls[0]
    # calculate advantages
    new = np.append(test_seg["new"], 0)
    vpred = np.append(test_seg["vpred"], 0)
    T = len(test_seg["rew"])
    test_seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = test_seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    test_seg["tdlamret"] = test_seg["adv"] + test_seg["vpred"]

    split_test = split_segments(test_seg, macrolen, num_subpolicies)
    return split_test

def split_segments(seg, macrolen, num_subpolicies):
    subpol_counts = []
    for i in range(num_subpolicies):
        subpol_counts.append(0)
    for macro_ac in seg["macro_ac"]:
        subpol_counts[macro_ac] += macrolen
    subpols = []
    for i in range(num_subpolicies):
        obs = np.array([seg["ob"][0] for _ in range(subpol_counts[i])])
        advs = np.zeros(subpol_counts[i], 'float32')
        tdlams = np.zeros(subpol_counts[i], 'float32')
        acs = np.array([seg["ac"][0] for _ in range(subpol_counts[i])])
        subpols.append({"ob": obs, "adv": advs, "tdlamret": tdlams, "ac": acs})
    subpol_counts = []
    for i in range(num_subpolicies):
        subpol_counts.append(0)
    for i in range(len(seg["ob"])):
        mac = seg["macro_ac"][int(i/macrolen)]
        subpols[mac]["ob"][subpol_counts[mac]] = seg["ob"][i]
        subpols[mac]["adv"][subpol_counts[mac]] = seg["adv"][i]
        subpols[mac]["tdlamret"][subpol_counts[mac]] = seg["tdlamret"][i]
        subpols[mac]["ac"][subpol_counts[mac]] = seg["ac"][i]
        subpol_counts[mac] += 1
    return subpols
