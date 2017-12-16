import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntObstaclesEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.mx = 0
        self.my = 20
        self.realgoal = np.array([0,1])
        mujoco_env.MujocoEnv.__init__(self, 'ant_obstacles.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = np.array([self.np_random.choice([0, 1]), self.np_random.choice([0, 1])])
        # 0 = obstacle. 1 = no obstacle.
        # self.realgoal = 0

    def _step(self, a):
        self.count += 1

        if self.count % 200 == 0:
            n_qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
            n_qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            n_qpos[:2] = self.data.qpos[:2,0]
            self.set_state(n_qpos, n_qvel)

        if np.sum(np.square(self.data.qpos[:2,0] - np.array([0,20]))) < 15*15:
            self.mx += np.sign(self.data.qpos[0,0] - self.mx)
            self.my += np.sign(self.data.qpos[1,0] - self.my)

        # print(np.square(self.data.qpos[:2] - np.array([0,20])))

        n_qpos = np.copy(self.data.qpos[:,0])
        n_qpos[-2:] = np.array([self.mx,self.my])
        self.set_state(n_qpos, self.data.qvel[:,0])
        self.do_simulation(a, self.frame_skip)

        reward = -np.square(np.sum(self.data.qpos[:2] - np.array([50,50]))) / 100000
        #
        # print(np.square(np.sum(self.data.qpos[:2] - np.array([50,50]))))

        # if np.square(np.sum(self.data.qpos[:2] - np.array([50,50]))) < 2000:
        #     reward = 1
        # else:
        #     reward = 0
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        # return np.concatenate([
        #     self.data.qpos.flat[2:],
        #     self.data.qvel.flat,
        # ])
        # return np.concatenate([
        #     self.data.qpos.flat,
        #     self.data.qvel.flat,
        # ])
        return np.concatenate([
            self.data.qpos.flat[:-2],
            self.data.qvel.flat[:-2],
        ])

    def reset_model(self):
        self.count = 0
        self.mx = 0
        self.my = 20
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.4
