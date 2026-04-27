import numpy as np

class Missile:
    def __init__(self, starting_point, velocity, acceleration):
        self.starting_point = starting_point
        self.position = starting_point
        self.velocity = velocity
        self.acceleration = acceleration

    def update(self, dt):
        self.position = self.position + self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity = self.velocity + self.acceleration * dt