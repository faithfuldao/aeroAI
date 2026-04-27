import gymnasium as gym
from gymnasium import spaces, Env
import random
import numpy as np
import os

CLOSER=1
INTERCEPTED=5
TARGET_HIT=-5
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
        self.state = [NOTHING] * (width*height*length)
        self.target = (random.randrange(width), random.randrange(height), 0)
        self.threat = (random.randrange(width), random.randrange(height), length-1)
        self.interceptor = (random.randrange(width), random.randrange(height), 0)
        self.reset()

        while self.interceptor == self.target:
            self.interceptor = (random.randrange(0, width), random.randrange(0, height), 0)
        
        self.state[self._to_idx(*self.target)] = TARGET
        self.state[self._to_idx(*self.threat)] = THREAT  
        self.state[self._to_idx(*self.interceptor)] = INTERCEPTOR
        
        pass

    #returns the exact grid index idx
    def _to_idx(self, x, y, z):
        return x + y * self.width + z * self.width * self.height

    def step(self, action):
        tx, ty, tz = self.threat
        tax, tay, taz = self.target

        if tx > tax: tx -= 1
        elif tx < tax: tx += 1

        if ty > tay: ty -= 1
        elif ty < tay: ty += 1

        tz -= 1 

        self.threat = (tx, ty, tz)
           

    def reset(self):    
        self.state = [NOTHING] * (self.width*self.height*self.length)
        self.target = (random.randrange(self.width), random.randrange(self.height), 0)
        self.threat = (random.randrange(self.width), random.randrange(self.height), self.length-1)
        self.interceptor = (random.randrange(self.width), random.randrange(self.height), 0)

        self.state[self._to_idx(*self.target)] = TARGET
        self.state[self._to_idx(*self.threat)] = THREAT  
        self.state[self._to_idx(*self.interceptor)] = INTERCEPTOR

        return np.array(self.state)
    
    def render(self):
        pretty_print(self.state, self.cumulative_reward)

if __name__ == "__main__":
    env = AirDefenseEnv(5, 5, 5)
    obs = env.reset()
    print(obs)