import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Obstacles(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.realgoal = np.array([-2, -2, -2, -2])
        mujoco_env.MujocoEnv.__init__(self, "obstacles.xml", 4)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        # just one jumping task
        # self.realgoal = np.array([0.125, -2, -2, -2])
        # a high jumping task
        # self.realgoal = np.array([0.425, -2, -2, -2])
        # a ducking task
        # self.realgoal = np.array([1.4, -2, -2, -2])
        # a repeated jumping task
        # self.realgoal = np.array([0.125, 0.125, 0.125, 0.125])
        # an easy running task
        # self.realgoal = np.array([-2, -2, -2, -2])
        # a distribution of easy jumping tasks
        # self.realgoal = np.array([self.np_random.choice([0.125, -2]), self.np_random.choice([0.125, -2]), self.np_random.choice([0.125, -2]), self.np_random.choice([0.125, -2])])
        # # a distribution of high jumping tasks
        # self.realgoal = np.array([self.np_random.choice([0.325, -2]), self.np_random.choice([0.325, -2]), -2, -2])
        # a distribution of medium jumping tasks
        # self.realgoal = np.array([self.np_random.choice([0.255, -2]), -2, -2, -2])
        # transfer
        self.realgoal = np.array([self.np_random.choice([0.225, -2]), self.np_random.choice([0.225, -2]), -2, -2])
        pass

    def _step(self, a):
        posbefore, heightbefore = self.model.data.qpos[0:2, 0]
        self.do_simulation(a, self.frame_skip)

        iq = np.copy(self.model.data.qpos)[:,0]
        iv = np.copy(self.model.data.qvel)[:,0]
        iq[-4:] = self.realgoal
        self.set_state(iq, iv)

        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 0.1
        reward = ((posafter - posbefore) / self.dt)
        heightrew = ((height - heightbefore) / self.dt)
        # print("rew %f, height %f" % (reward, heightrew))
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        # if not done:
        #     reward += alive_bonus
        reward += heightrew

        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos[:-4]
        qvel = self.model.data.qvel[:-4]
        # print(qpos)
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        iq = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        iv = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        iq[-4:] = self.realgoal
        # iv[-4:] = 0
        self.set_state(iq, iv)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
