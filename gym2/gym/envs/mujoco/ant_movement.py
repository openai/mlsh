import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntMovementEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.realgoal = np.array([1,3])
        mujoco_env.MujocoEnv.__init__(self, 'ant_v2.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = np.array([self.np_random.choice([1, 3])])
        # 0 = obstacle. 1 = no obstacle.

    def _step(self, a):
        # print(self.data.qpos.shape)
        xposbefore = self.data.qpos[0,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[1,0]
        yposbefore = self.data.qpos[1,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[0,0]

        self.do_simulation(a, self.frame_skip)

        xposafter = self.data.qpos[0,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[1,0]
        yposafter = self.data.qpos[1,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[0,0]

        forward_reward = (xposafter - xposbefore)/self.dt
        # if self.realgoal[0] == 1 or self.realgoal[0] == 3:
            # forward_reward = forward_reward * -1
        side_reward = np.abs(yposafter) * 0.5
        ctrl_cost = .1 * np.square(a).sum()
        reward = forward_reward - ctrl_cost - side_reward
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(forward_reward=forward_reward, ctrl_cost=ctrl_cost, side_reward=side_reward)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.2
