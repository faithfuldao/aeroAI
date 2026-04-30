import os
import glob
from air_defense_env import AirDefenseEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

N_ENVS = 8
TIMESTEPS = 1_000_000
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PREFIX = "air_defense_sac"


def find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*_steps.zip"))
    if not files:
        return None, 0
    latest = max(files, key=lambda f: int(f.split("_steps.zip")[0].split("_")[-1]))
    steps = int(latest.split("_steps.zip")[0].split("_")[-1])
    return latest, steps


if __name__ == '__main__':
    env = make_vec_env(lambda: AirDefenseEnv(), n_envs=N_ENVS)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_PREFIX,
        save_replay_buffer=True,
    )

    latest_checkpoint, steps_done = find_latest_checkpoint()

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint} ({steps_done} steps done)")
        model = SAC.load(latest_checkpoint, env=env, tensorboard_log="./tb_logs")
        replay_buffer_path = latest_checkpoint.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            print("Replay buffer restored.")
        remaining = TIMESTEPS - steps_done
    else:
        print("No checkpoint found, starting fresh.")
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
            train_freq=8,
            gradient_steps=4,
            tensorboard_log="./tb_logs",
        )
        remaining = TIMESTEPS

    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_callback,
        reset_num_timesteps=latest_checkpoint is None,
    )
    model.save("./air_defense_sac")
    print("model saved")
