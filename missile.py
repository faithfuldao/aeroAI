import numpy as np


class Missile:
    SPEED = 500.0           # m/s (~Mach 1.5)
    MAX_TURN_ACCEL = 15 * 9.81  # 15g steering capability
    N = 4.0                 # proportional navigation constant

    def __init__(self, position, target_pos):
        self.position = np.array(position, dtype=np.float64)
        self.target_pos = np.array(target_pos, dtype=np.float64)

        direction = self.target_pos - self.position
        direction /= np.linalg.norm(direction)
        self.velocity = direction * self.SPEED

        self._prev_los_hat = None

    def update(self, dt):
        to_target = self.target_pos - self.position
        dist = np.linalg.norm(to_target)
        if dist < 1.0:
            return

        los_hat = to_target / dist

        if self._prev_los_hat is None:
            los_rate = np.zeros(3)
        else:
            los_rate = (los_hat - self._prev_los_hat) / dt
        self._prev_los_hat = los_hat.copy()

        closing_speed = max(np.dot(-self.velocity, los_hat), 0.0)
        accel = self.N * closing_speed * los_rate
        accel[2] -= 9.81  # gravity

        self.velocity += accel * dt

        speed = np.linalg.norm(self.velocity)
        if speed > self.SPEED:
            self.velocity = self.velocity / speed * self.SPEED

        self.position += self.velocity * dt
