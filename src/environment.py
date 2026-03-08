import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import random

class CarRacingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):

        super(CarRacingEnv, self).__init__()

        self.render_mode = render_mode

        self.width = 800
        self.height = 600

        self.num_rays = 8
        self.max_steps = 1000

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_rays + 1,),
            dtype=np.float32
        )

        self.position = np.array([400.0, 300.0])
        self.velocity = 0.0
        self.angle = 0.0

        self.step_count = 0

        self.walls = []
        self.load_track()

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def load_track(self):

        track_path = os.path.join(os.path.dirname(__file__), "..", "tracks", "track_1.txt")

        with open(track_path, "r") as f:
            for line in f.readlines():
                x1, y1, x2, y2 = map(float, line.strip().split())
                self.walls.append((x1, y1, x2, y2))

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.position = np.array([400.0, 300.0])
        self.velocity = 0.0
        self.angle = 0.0
        self.step_count = 0

        observation = self.get_observation()

        return observation, {}

    def step(self, action):

        if action == 1:
            self.velocity += 0.2

        if action == 2:
            self.velocity -= 0.2

        if action == 3:
            self.angle -= 5

        if action == 4:
            self.angle += 5

        self.velocity = np.clip(self.velocity, -2, 4)

        rad = np.deg2rad(self.angle)

        self.position[0] += np.cos(rad) * self.velocity
        self.position[1] += np.sin(rad) * self.velocity

        self.step_count += 1

        observation = self.get_observation()

        reward = 0.1

        terminated = False

        if self.check_collision():
            reward = -10
            terminated = True

        truncated = self.step_count >= self.max_steps

        return observation, reward, terminated, truncated, {}

    def get_observation(self):

        rays = np.ones(self.num_rays)

        velocity = np.array([self.velocity / 4])

        obs = np.concatenate((rays, velocity)).astype(np.float32)

        return obs

    def check_collision(self):

        x, y = self.position

        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True

        return False

    def render(self):

        if self.render_mode != "human":
            return

        self.screen.fill((30, 30, 30))

        for wall in self.walls:
            pygame.draw.line(self.screen, (255,255,255), (wall[0],wall[1]), (wall[2],wall[3]), 3)

        pygame.draw.circle(self.screen, (0,255,0), self.position.astype(int), 8)

        pygame.display.flip()

        self.clock.tick(self.metadata["render_fps"])

    def close(self):

        if self.render_mode == "human":
            pygame.quit()