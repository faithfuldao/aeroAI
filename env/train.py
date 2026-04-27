from air_defense_env import AirDefenseEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = AirDefenseEnv(10, 10, 10)
check_env(env)
print("env is valid")

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

model.save("air_defense_ppo")
print("model saved")