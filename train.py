import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environment import CarRacingEnv


class TrainingLogger(BaseCallback):

    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.mean_rewards = []

    def _on_step(self):

        if len(self.model.ep_info_buffer) > 0:

            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])

            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(float(mean_reward))

        return True


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


env = CarRacingEnv()

ppo_params = config["ppo_params"]

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=ppo_params["learning_rate"],
    n_steps=ppo_params["n_steps"],
    batch_size=ppo_params["batch_size"],
    gamma=ppo_params["gamma"],
    verbose=1
)


logger = TrainingLogger()

total_timesteps = config["training"]["total_timesteps"]

model.learn(total_timesteps=total_timesteps, callback=logger)


model.save("models/ppo_car_agent")


log_data = {
    "timesteps": logger.timesteps,
    "mean_rewards": logger.mean_rewards
}

with open("results/training_log.json", "w") as f:
    json.dump(log_data, f)


plt.plot(logger.timesteps, logger.mean_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.savefig("results/reward_curve.png")
plt.close()