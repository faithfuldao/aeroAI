import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from gymnasium import spaces, Env

from missile import Missile
from interceptor import Interceptor

KILL_RADIUS = 50.0      # meters — interception if closer than this
MAX_STEPS   = 120       # ~60 seconds at dt=0.5s
ZONE_XY     = 20000.0   # 20 km engagement area
ZONE_Z      = 10000.0   # 10 km altitude ceiling


class AirDefenseEnv(Env):
    def __init__(self):
        self.dt = 0.5  # seconds per simulation step

        # Agent commands 3D normalized acceleration [-1, 1] on each axis
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: relative position (3) + interceptor velocity (3) + missile velocity (3)
        # All normalized to roughly [-1, 1]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(9,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        rel = self.missile.position - self.interceptor.position
        return np.array([
            rel[0] / ZONE_XY,
            rel[1] / ZONE_XY,
            rel[2] / ZONE_Z,
            self.interceptor.velocity[0] / Interceptor.MAX_SPEED,
            self.interceptor.velocity[1] / Interceptor.MAX_SPEED,
            self.interceptor.velocity[2] / Interceptor.MAX_SPEED,
            self.missile.velocity[0] / Missile.SPEED,
            self.missile.velocity[1] / Missile.SPEED,
            self.missile.velocity[2] / Missile.SPEED,
        ], dtype=np.float32)

    def step(self, action):
        prev_dist = np.linalg.norm(self.missile.position - self.interceptor.position)

        self.missile.update(self.dt)
        self.interceptor.update(action, self.dt)
        self.steps += 1

        curr_dist = np.linalg.norm(self.missile.position - self.interceptor.position)

        # Shaping: small reward for closing distance each step
        reward = (prev_dist - curr_dist) * 0.005

        done = False
        truncated = False

        if curr_dist < KILL_RADIUS:
            reward = 100.0
            done = True
        elif self.missile.position[2] <= 0.0:
            reward = -100.0
            done = True
        elif self.steps >= MAX_STEPS:
            reward = -50.0
            truncated = True

        return self._get_obs(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        self.steps = 0

        self.target_pos = np.array([
            random.uniform(2000, ZONE_XY - 2000),
            random.uniform(2000, ZONE_XY - 2000),
            0.0
        ])

        missile_start = np.array([
            random.uniform(0, ZONE_XY),
            random.uniform(0, ZONE_XY),
            ZONE_Z
        ])

        self.missile = Missile(missile_start, self.target_pos)

        interceptor_start = self.target_pos + np.array([
            random.uniform(-500, 500),
            random.uniform(-500, 500),
            0.0
        ])
        self.interceptor = Interceptor(interceptor_start)

        return self._get_obs(), {}

    def render(self):
        dist = np.linalg.norm(self.missile.position - self.interceptor.position)
        print(
            f"step {self.steps:>3} | "
            f"missile {self.missile.position.astype(int)} | "
            f"interceptor {self.interceptor.position.astype(int)} | "
            f"dist {dist:.0f} m"
        )
