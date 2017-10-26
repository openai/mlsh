import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model):
    mass = model.body_mass[:-1]
    xpos = model.data.xipos[:-1]
    # print(mass.shape)
    # print(xpos.shape)
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidSeqEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.current = 0
        self.realgoal = np.array([1,1,1,1])
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_course.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = self.np_random.choice([0, 1], size=(4,))
        self.realgoal = np.array([0,0,0,0])
        # 0 = crawl. 1 = walk.

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos[2:-1].flat,
                               data.qvel[:-1].flat,
                               data.cinert[:-1].flat,
                               data.cvel[:-1].flat,
                               data.qfrc_actuator[:-1].flat,
                               data.cfrc_ext[:-1].flat])
        # return np.concatenate([data.qpos.flat[2:],
        #                        data.qvel.flat,
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat])

    def _step(self, a):
        if self.count % 250 == 0:
            self.current = self.realgoal[int(self.count / 250)];
            # print("current is %d" % self.current)

        self.count += 1
        pos_before = mass_center(self.model)
        height_before = self.model.data.qpos[2][0]
        self.do_simulation(a, self.frame_skip)
        height_after = self.model.data.qpos[2][0]


        iq = np.copy(self.model.data.qpos)[:,0]
        iv = np.copy(self.model.data.qvel)[:,0]
        iq[-1] = 30
        self.set_state(iq, iv)

        if self.current == 0: # crawling
            pos_after = mass_center(self.model)
            alive_bonus = 5.0
            data = self.model.data
            lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)

            reward = 0 - quad_ctrl_cost - quad_impact_cost

            qpos = self.model.data.qpos
            if bool((qpos[2] > 1.0)):
                reward += (height_before - height_after) / self.model.opt.timestep
            else:
                reward += alive_bonus + lin_vel_cost

            done = False
        elif self.current == 1: # walking
            pos_after = mass_center(self.model)
            alive_bonus = 5.0
            data = self.model.data
            lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)

            reward = 0 - quad_ctrl_cost - quad_impact_cost

            qpos = self.model.data.qpos
            if bool((qpos[2] < 1.0)):
                reward += (height_after - height_before) / self.model.opt.timestep
            else:
                reward += alive_bonus + lin_vel_cost

            # done = bool((qpos[2] < 1.0))
            done = False

        # print(qpos[2])
        # if self.count % 10 == 0:
        #     print(reward)
            # print((height_before - height_after) / self.model.opt.timestep)

        return self._get_obs(), reward, done, {}

    def reset_model(self):
        self.count = 0
        self.current = 0
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 9 * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
