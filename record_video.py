import os
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

from src.environment import CarRacingEnv


video_folder = "results"

env = CarRacingEnv(render_mode="rgb_array")

env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda e: True
)

model = PPO.load("models/ppo_car_agent")

obs, _ = env.reset()

done = False

while not done:

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

env.close()


video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

if len(video_files) > 0:

    src = os.path.join(video_folder, video_files[0])

    dst = os.path.join(video_folder, "agent_demonstration.mp4")

    os.replace(src, dst)