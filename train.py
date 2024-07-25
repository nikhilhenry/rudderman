from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import argparse
from env import KarmanVortexStreetEnv
from utils import VideoRecorderCallback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of the experiment to run")
parser.add_argument("dir", help="path to save models")
args = parser.parse_args()

print(f"Running experiment {args.name}")

# Instantiate the env
env = TimeLimit(KarmanVortexStreetEnv(), max_episode_steps=5000)
# Train the agent
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
video_recoder = VideoRecorderCallback(
    Monitor(KarmanVortexStreetEnv(render_mode="rgb_array")),
    render_freq=5000,
)
model.learn(int(2e5), tb_log_name=args.name, progress_bar=True, callback=video_recoder)

save_path = Path(args.dir)
save_path.mkdir(parents=True, exist_ok=True)

model.save(save_path / args.name)
