import gymnasium as gym
from gymnasium import spaces, Env
import random
import numpy as np
import os
from stable_baselines3 import PPO

INTERCEPTED=10
TARGET_HIT=-10
NOTHING=0
TARGET=1 
THREAT=2
INTERCEPTOR=3

#formula for 3D dimension location: 
#FlatIndex => idx= x+y*width+z*width*height


class AirDefenseEnv(Env):
    def __init__(self, width, height, length):
        self.width = width
        self.height = height
        self.length = length
        self.cumulative_reward = 0

        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(width * height * length,),
            dtype=np.int16
        )
        
        self.reset()
        
        pass

    #returns the exact grid index idx
    def _to_idx(self, x, y, z):
        return x + y * self.width + z * self.width * self.height
    
    def _distance(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def step(self, action):
        prev_distance = self._distance(self.interceptor, self.threat)
        tx, ty, tz = self.threat
        tax, tay, taz = self.target

        if tx > tax: tx -= 1
        elif tx < tax: tx += 1

        if ty > tay: ty -= 1
        elif ty < tay: ty += 1

        tz -= 1 

        self.threat = (tx, ty, tz)

        ix, iy, iz = self.interceptor

        if action == 0: ix -= 1   # left
        elif action == 1: ix += 1  # righ
        elif action == 2: iy -= 1  # forward
        elif action == 3: iy += 1  # backward
        elif action == 4: iz -= 1  # descend
        elif action == 5: iz += 1  # climb

        ix = max(0, min(self.width - 1, ix))
        iy = max(0, min(self.height - 1, iy))
        iz = max(0, min(self.length - 1, iz))

        self.interceptor = (ix, iy, iz)

        if tz == 0:
            reward = TARGET_HIT
            done = True
        elif self.interceptor == self.threat:
            reward = INTERCEPTED
            done = True
        else:
            reward = 0
            done = False

        self.state = [NOTHING] * (self.width * self.height * self.length)
        self.state[self._to_idx(*self.target)] = TARGET
        self.state[self._to_idx(*self.threat)] = THREAT
        self.state[self._to_idx(*self.interceptor)] = INTERCEPTOR

        self.cumulative_reward += reward
        observation = np.array(self.state, dtype=np.int16)
        truncated = False
        info = {}
        return observation, reward, done, truncated, info
           

    def reset(self, seed=None, options=None):
        self.state = [NOTHING] * (self.width*self.height*self.length)
        self.target = (random.randrange(self.width), random.randrange(self.height), 0)
        self.threat = (random.randrange(self.width), random.randrange(self.height), self.length-1)
        ix = max(0, min(self.width - 1, self.target[0] + random.randint(-2, 2)))
        iy = max(0, min(self.height - 1, self.target[1] + random.randint(-2, 2)))
        self.interceptor = (ix, iy, 0)

        self.state[self._to_idx(*self.target)] = TARGET
        self.state[self._to_idx(*self.threat)] = THREAT  
        self.state[self._to_idx(*self.interceptor)] = INTERCEPTOR

        return np.array(self.state, dtype=np.int16), {}
    
    def render(self):
        pretty_print(self.state, self.cumulative_reward)

if __name__ == "__main__":
    env = AirDefenseEnv(5, 5, 5)
    obs, _ = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"reward: {reward}, done: {done}")
        if done:
            obs, _ = env.reset()  