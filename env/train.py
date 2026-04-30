from air_defense_env import AirDefenseEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

N_ENVS = 8
TIMESTEPS = 1_000_000

env = make_vec_env(lambda: AirDefenseEnv(10, 10, 10), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints",
    name_prefix="air_defense_ppo",
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    tensorboard_log="./tb_logs",
)

model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

model.save("./air_defense_ppo")
print("model saved")
