import numpy as np


class Interceptor:
    MAX_SPEED = 900.0        # m/s (~Mach 2.6)
    MAX_ACCEL = 20 * 9.81    # 20g = 196 m/s²

    def __init__(self, position):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)

    def update(self, action, dt):
        accel = np.array(action, dtype=np.float64) * self.MAX_ACCEL
        self.velocity += accel * dt

        speed = np.linalg.norm(self.velocity)
        if speed > self.MAX_SPEED:
            self.velocity = self.velocity / speed * self.MAX_SPEED

        self.position += self.velocity * dt
        self.position[2] = max(0.0, self.position[2])  # floor at ground
