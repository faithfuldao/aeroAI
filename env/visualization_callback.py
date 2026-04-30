import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from air_defense_env import AirDefenseEnv

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

            tx, ty, tz = self.env.threat
            ix, iy, iz = self.env.interceptor
            gx, gy, gz = self.env.target

            self.ax.scatter(gx, gy, gz, color='blue', s=100, label='target')
            self.ax.scatter(ix, iy, iz, color='green', s=100, label='interceptor')
            self.ax.scatter(tx, ty, tz, color='red', s=100, label='threat')

            self.ax.set_xlim(0, self.env.width)
            self.ax.set_ylim(0, self.env.height)
            self.ax.set_zlim(0, self.env.length)

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z (altitude)')
            self.ax.legend()
            self.ax.set_title(f"step {self.n_calls} | reward: {self.env.cumulative_reward}")

            plt.pause(0.5) 

        return True 


if __name__ == "__main__":
    env = AirDefenseEnv(10, 10, 10)

    callback = VisualizationCallback(env, update_freq=100)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000, callback=callback)
    model.save("air_defense_ppo")