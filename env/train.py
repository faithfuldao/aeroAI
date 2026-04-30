from air_defense_env import AirDefenseEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

N_ENVS = 8
TIMESTEPS = 2_000_000

if __name__ == '__main__':
    env = make_vec_env(lambda: AirDefenseEnv(), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints",
        name_prefix="air_defense_sac",
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log="./tb_logs",
    )

    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    model.save("./air_defense_sac")
    print("model saved")
