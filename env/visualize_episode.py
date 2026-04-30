import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from air_defense_env import AirDefenseEnv
import os

STEP_DELAY = 0.2   # seconds between frames — lower = faster
EPISODE_PAUSE = 0.5  # pause on final frame before next episode
NUM_EPISODES = 0   # 0 = run forever until window is closed


def run_episode(model, env):
    obs, _ = env.reset()
    done = False
    history = {
        "threat": [env.threat],
        "interceptor": [env.interceptor],
        "target": env.target,
    }
    steps = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        history["threat"].append(env.threat)
        history["interceptor"].append(env.interceptor)
        steps.append({"reward": reward, "done": done, "action": int(action)})

    return history, steps


def animate_episode(ax, history, steps, env_size, episode_num, stats):
    ACTION_NAMES = ["left", "right", "forward", "backward", "descend", "climb"]
    w, h, l = env_size
    total_steps = len(steps)

    threat_xs, threat_ys, threat_zs = [], [], []
    inter_xs, inter_ys, inter_zs = [], [], []

    last_step = steps[-1]
    intercepted = history["interceptor"][-1] == history["threat"][-1]
    outcome_label = "INTERCEPTED" if intercepted else "TARGET HIT"

    print(f"\n--- Episode {episode_num} ({total_steps} steps) ---")

    for i, step in enumerate(steps):
        if not plt.fignum_exists(ax.figure.number):
            return False  # window was closed

        ax.cla()

        t = history["threat"][i + 1]
        it = history["interceptor"][i + 1]
        threat_xs.append(t[0]); threat_ys.append(t[1]); threat_zs.append(t[2])
        inter_xs.append(it[0]); inter_ys.append(it[1]); inter_zs.append(it[2])

        ax.plot(threat_xs, threat_ys, threat_zs, "r--", alpha=0.35, linewidth=1)
        ax.plot(inter_xs, inter_ys, inter_zs, "g--", alpha=0.35, linewidth=1)

        ax.scatter(*history["target"], color="blue", s=120, zorder=5, label="Target")
        ax.scatter(*history["threat"][i + 1], color="red", s=120, zorder=5, label="Threat")
        ax.scatter(*history["interceptor"][i + 1], color="green", s=120, zorder=5, label="Interceptor")

        ax.scatter(*history["threat"][0], color="red", s=40, alpha=0.2)
        ax.scatter(*history["interceptor"][0], color="green", s=40, alpha=0.2)

        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(0, l)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (altitude)")
        ax.legend(loc="upper left")

        outcome = f"  *** {outcome_label} ***" if step["done"] else ""
        ax.set_title(
            f"Episode {episode_num} | Step {i + 1}/{total_steps} | "
            f"action: {ACTION_NAMES[step['action']]} | reward: {step['reward']}{outcome}\n"
            f"wins: {stats['wins']}  losses: {stats['losses']}"
        )

        print(
            f"  step {i+1:>3} | {ACTION_NAMES[step['action']]:>8} | "
            f"threat {history['threat'][i+1]} | "
            f"interceptor {history['interceptor'][i+1]} | "
            f"reward {step['reward']}"
            + (outcome if outcome else "")
        )

        delay = EPISODE_PAUSE if step["done"] else STEP_DELAY
        plt.pause(delay)

    return True


if __name__ == "__main__":
    env = AirDefenseEnv(10, 10, 10)

    model_path = os.path.join(os.path.dirname(__file__), "air_defense_ppo")
    if not os.path.exists(model_path + ".zip"):
        print("No saved model found — using random policy.")
        model = PPO("MlpPolicy", env, verbose=0)
    else:
        print(f"Loading model from {model_path}.zip")
        model = PPO.load(model_path, env=env)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()

    stats = {"wins": 0, "losses": 0}
    episode = 0

    while NUM_EPISODES == 0 or episode < NUM_EPISODES:
        episode += 1
        history, steps = run_episode(model, env)

        intercepted = history["interceptor"][-1] == history["threat"][-1]
        if intercepted:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        still_open = animate_episode(ax, history, steps, (env.width, env.height, env.length), episode, stats)
        if not still_open:
            break

    print(f"\n=== Done | {stats['wins']} wins / {stats['losses']} losses ===")
    plt.ioff()
    plt.show()
