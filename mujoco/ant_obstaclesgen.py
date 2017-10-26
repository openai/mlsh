import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntObstaclesGenEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.realgoal = 0
        mujoco_env.MujocoEnv.__init__(self, 'ant_obstacles_gen.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = self.np_random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # 0 = obstacle. 1 = no obstacle.
        self.realgoal = 6

    def _step(self, a):
        self.count += 1

        if self.count % 200 == 0:
            n_qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
            n_qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            n_qpos[:2] = self.data.qpos[:2,0]
            n_qpos[-11:] = self.data.qpos[-11:,0]
            self.set_state(n_qpos, n_qvel)

        goal = np.array([8, 24])
        if self.realgoal == 0:
            goal = np.array([8, 24])
        if self.realgoal == 1:
            goal = np.array([8, -24])
        if self.realgoal == 2:
            goal = np.array([24, 24])
        if self.realgoal == 3:
            goal = np.array([24, -24])
        if self.realgoal == 4:
            goal = np.array([48, 0])

        if self.realgoal == 5:
            goal = np.array([40, 24])
        if self.realgoal == 6:
            goal = np.array([40, -24])
        if self.realgoal == 7:
            goal = np.array([32, 16])
        if self.realgoal == 8:
            goal = np.array([32, -16])

        # reward = -np.sum(np.square(self.data.qpos[:2,0] - goal)) / 100000

        xposbefore = self.data.qpos[0,0]
        yposbefore = self.data.qpos[1,0]

        self.do_simulation(a, self.frame_skip)

        xposafter = self.data.qpos[0,0]
        yposafter = self.data.qpos[1,0]

        if xposbefore < goal[0]:
            forward_reward = (xposafter - xposbefore)/self.dt
        else:
            forward_reward = -1*(xposafter - xposbefore)/self.dt
        if yposbefore < goal[1]:
            forward_reward += (yposafter - yposbefore)/self.dt
        else:
            forward_reward += -1*(yposafter - yposbefore)/self.dt

        ctrl_cost = .1 * np.square(a).sum()
        reward = forward_reward - ctrl_cost

        # print(reward)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:-11],
            self.data.qvel.flat[:-11],
            # self.data.qpos.flat,
            # self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        # self.realgoal = 4
        if self.realgoal == 0:
            qpos[-11:] = np.array([80,0,0,80,0,0,0,0,0, 8, 24])
        if self.realgoal == 1:
            qpos[-11:] = np.array([0,0,0,80,0,0,80,0,0, 8, -24])
        if self.realgoal == 2:
            qpos[-11:] = np.array([0,80,0,80,80,0,0,0,0, 24, 24])
        if self.realgoal == 3:
            qpos[-11:] = np.array([0,0,0,80,80,0,0,80,0, 24, -24])
        if self.realgoal == 4:
            qpos[-11:] = np.array([0,0,0,80,80,80,0,0,0, 48, 0])

        if self.realgoal == 5:
            qpos[-11:] = np.array([0,0,80,80,80,80,0,0,0, 40, 24])
        if self.realgoal == 6:
            qpos[-11:] = np.array([0,0,0,80,80,80,0,0,80, 40, -24])
        if self.realgoal == 7:
            qpos[-11:] = np.array([80,80,0,80,0,0,0,0,0, 32, 16])
        if self.realgoal == 8:
            qpos[-11:] = np.array([0,0,0,80,0,0,80,80,0, 32, -16])

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.6
