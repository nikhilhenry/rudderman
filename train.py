from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from env import KarmanVortexStreetEnv

# Instantiate the env
env = TimeLimit(KarmanVortexStreetEnv(), max_episode_steps=1500)

# Train the agent
model = DDPG("MlpPolicy", env, verbose=1,tensorboard_log="./logs/")
model.learn(5000,tb_log_name="first_try",progress_bar=True)
