import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MovementBandits-v0',
    entry_point='test_envs.envs:MovementBandits',
    timestep_limit=50,
)

register(
    id='KeyDoor-v0',
    entry_point='test_envs.envs:KeyDoor',
    timestep_limit=100,
)

register(
    id='Allwalk-v0',
    entry_point='test_envs.envs:Allwalk',
    timestep_limit=50,
)

register(
    id='Fourrooms-v0',
    entry_point='test_envs.envs:Fourrooms',
    timestep_limit=100,
    reward_threshold = 1,
)
