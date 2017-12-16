import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model):
    mass = model.body_mass[:-1]
    xpos = model.data.xipos[:-1]
    # print(mass.shape)
    # print(xpos.shape)
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.realgoal = 1
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = self.np_random.choice([0, 1])
        # 0 = obstacle. 1 = no obstacle.
        # self.realgoal = 0

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos[:-1].flat[2:],
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
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)

        iq = np.copy(self.model.data.qpos)[:,0]
        iv = np.copy(self.model.data.qvel)[:,0]
        iq[-1] = 0
        if self.realgoal == 1:
            iq[-1] = 30
        self.set_state(iq, iv)

        # pos_after = mass_center(self.model)
        # alive_bonus = 5.0
        # data = self.model.data
        # lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        # quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        # quad_impact_cost = min(quad_impact_cost, 10)
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        # qpos = self.model.data.qpos
        # if self.realgoal == 0:
        #     done = bool((qpos[2] < 0.1) or (qpos[2] > 2.0))
        # elif self.realgoal == 1:
        #     done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        if self.realgoal == 0:
            pos_after = mass_center(self.model)
            alive_bonus = 5.0
            data = self.model.data
            lin_vel_cost = 1.5 * (pos_after - pos_before) / self.model.opt.timestep
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            reward = 0 - lin_vel_cost - quad_ctrl_cost - quad_impact_cost
            done = False
        elif self.realgoal == 1:
            pos_after = mass_center(self.model)
            alive_bonus = 5.0
            data = self.model.data
            lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            reward = 0 - quad_ctrl_cost - quad_impact_cost
            qpos = self.model.data.qpos
            if not bool((qpos[2] < 1.0)):
                reward += alive_bonus + lin_vel_cost
            done = bool((qpos[2] < 1.0))
            # done = False

        # print(qpos[2])

        return self._get_obs(), reward, done, {}

    def reset_model(self):
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
