from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv

from gym.envs.mujoco.swimmer_bandits import SwimmerBanditsEnv
from gym.envs.mujoco.ant_bandits import AntBanditsEnv
from gym.envs.mujoco.obstacles import Obstacles

from gym.envs.mujoco.ant_movement import AntMovementEnv
from gym.envs.mujoco.ant_obstacles import AntObstaclesEnv
from gym.envs.mujoco.ant_obstaclesbig import AntObstaclesBigEnv
from gym.envs.mujoco.ant_obstaclesgen import AntObstaclesGenEnv
from gym.envs.mujoco.humanoid_course import HumanoidCourseEnv
from gym.envs.mujoco.humanoid_seq import HumanoidSeqEnv
