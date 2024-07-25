from stable_baselines3 import DDPG
from gymnasium.wrappers import TimeLimit
from env import KarmanVortexStreetEnv
from utils import VideoRecorderCallback

# Instantiate the env
env = TimeLimit(KarmanVortexStreetEnv(), max_episode_steps=1500)

# Train the agent
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
video_recoder = VideoRecorderCallback(KarmanVortexStreetEnv(render_mode="rgb_array"),
    render_freq=1000,
)
model.learn(5000, tb_log_name="first_try", progress_bar=True, callback=video_recoder)
