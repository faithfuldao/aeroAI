import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from gymnasium import spaces, Env

from missile import Missile
from interceptor import Interceptor

KILL_RADIUS = 150.0      # meters — interception if closer than this
MAX_STEPS   = 120       # ~60 seconds at dt=0.5s
ZONE_XY     = 20000.0   # 20 km engagement area
ZONE_Z      = 10000.0   # 10 km altitude ceiling


class AirDefenseEnv(Env):
    def __init__(self):
        self.dt = 0.5 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(12,), dtype=np.float32)

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
            self.missile.target_pos[0] / ZONE_XY,   
            self.missile.target_pos[1] / ZONE_XY,   
            self.missile.target_pos[2] / ZONE_Z, 
        ], dtype=np.float32)

    def step(self, action):
        prev_dist = np.linalg.norm(self.missile.position - self.interceptor.position)

        self.missile.update(self.dt)
        self.interceptor.update(action, self.dt)
        self.steps += 1

        curr_dist = np.linalg.norm(self.missile.position - self.interceptor.position)
        self._min_dist = min(self._min_dist, curr_dist)

        # lead the missile by an amount proportional to current distance:
        # far away → long lead; close in → aim almost directly at it
        lookahead = min(curr_dist / 1000.0, 3.0)
        predicted_pos = self.missile.position + self.missile.velocity * lookahead
        to_predicted = predicted_pos - self.interceptor.position
        dist_to_predicted = np.linalg.norm(to_predicted) + 1e-6
        approach_vel = np.dot(self.interceptor.velocity, to_predicted / dist_to_predicted)
        reward = approach_vel * 0.01
        reward -= 0.1

        done = False
        truncated = False

        if curr_dist < KILL_RADIUS:
            reward += 100.0
            done = True
        elif self.missile.position[2] <= 0.0:
            # closest-approach bonus so the agent gets signal even on failures
            reward += max(0.0, 1.0 - self._min_dist / 5000.0) * 30.0
            reward -= 100.0
            done = True
        elif self.steps >= MAX_STEPS:
            reward += max(0.0, 1.0 - self._min_dist / 5000.0) * 30.0
            reward -= 50.0
            truncated = True

        return self._get_obs(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        self.steps = 0
        self._min_dist = float('inf')

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

        # start with velocity pointing toward the missile so the agent
        # doesn't waste the first several steps just accelerating from rest
        toward_missile = self.missile.position - self.interceptor.position
        toward_missile /= np.linalg.norm(toward_missile)
        self.interceptor.velocity = toward_missile * 200.0

        return self._get_obs(), {}

    def render(self):
        dist = np.linalg.norm(self.missile.position - self.interceptor.position)
        print(
            f"step {self.steps:>3} | "
            f"missile {self.missile.position.astype(int)} | "
            f"interceptor {self.interceptor.position.astype(int)} | "
            f"dist {dist:.0f} m"
        )
