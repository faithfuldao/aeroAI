import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from air_defense_env import AirDefenseEnv, ZONE_XY, ZONE_Z


class VisualizationCallback(BaseCallback):
    def __init__(self, env, update_freq=100):
        super().__init__()
        self.env = env
        self.update_freq = update_freq

        self.fig = plt.figure()
        self.fig.suptitle("3D Air Defense - Live Training")
        self.ax = self.fig.add_subplot(projection='3d')
        plt.ion()

    def _on_step(self):
        if self.n_calls % self.update_freq == 0:
            self.ax.cla()

            mp = self.env.missile.position
            ip = self.env.interceptor.position
            tp = self.env.target_pos
            dist = np.linalg.norm(mp - ip)

            self.ax.scatter(*tp, color='blue', s=100, label='Target')
            self.ax.scatter(*ip, color='green', s=100, label='Interceptor')
            self.ax.scatter(*mp, color='red', s=100, label='Missile')

            self.ax.set_xlim(0, ZONE_XY)
            self.ax.set_ylim(0, ZONE_XY)
            self.ax.set_zlim(0, ZONE_Z)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Altitude (m)')
            self.ax.legend()
            self.ax.set_title(f"step {self.n_calls} | dist {dist:.0f} m")

            plt.pause(0.01)

        return True


if __name__ == "__main__":
    from stable_baselines3 import SAC

    env = AirDefenseEnv()
    callback = VisualizationCallback(env, update_freq=100)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000, callback=callback)
    model.save("air_defense_sac")
