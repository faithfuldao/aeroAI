import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from air_defense_env import AirDefenseEnv
import os

STEP_DELAY = 0.4  # seconds between frames — lower = faster


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


def animate(history, steps, env_size):
    ACTION_NAMES = ["left", "right", "forward", "backward", "descend", "climb"]
    w, h, l = env_size

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()

    threat_xs, threat_ys, threat_zs = [], [], []
    inter_xs, inter_ys, inter_zs = [], [], []

    tx, ty, tz = history["target"]
    total_steps = len(steps)

    for i, step in enumerate(steps):
        ax.cla()

        # accumulate trails
        t = history["threat"][i + 1]
        it = history["interceptor"][i + 1]
        threat_xs.append(t[0]); threat_ys.append(t[1]); threat_zs.append(t[2])
        inter_xs.append(it[0]); inter_ys.append(it[1]); inter_zs.append(it[2])

        # trails
        ax.plot(threat_xs, threat_ys, threat_zs, "r--", alpha=0.35, linewidth=1)
        ax.plot(inter_xs, inter_ys, inter_zs, "g--", alpha=0.35, linewidth=1)

        # current positions
        ax.scatter(*history["target"], color="blue", s=120, zorder=5, label="Target")
        ax.scatter(*history["threat"][i + 1], color="red", s=120, zorder=5, label="Threat")
        ax.scatter(*history["interceptor"][i + 1], color="green", s=120, zorder=5, label="Interceptor")

        # starting ghost positions
        ax.scatter(*history["threat"][0], color="red", s=40, alpha=0.2)
        ax.scatter(*history["interceptor"][0], color="green", s=40, alpha=0.2)

        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(0, l)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (altitude)")
        ax.legend(loc="upper left")

        outcome = ""
        if step["done"]:
            if history["interceptor"][i + 1] == history["threat"][i + 1]:
                outcome = "  *** INTERCEPTED ***"
            else:
                outcome = "  *** TARGET HIT ***"

        ax.set_title(
            f"Step {i + 1}/{total_steps} | action: {ACTION_NAMES[step['action']]} | "
            f"reward: {step['reward']}{outcome}"
        )

        print(
            f"step {i+1:>3} | {ACTION_NAMES[step['action']]:>8} | "
            f"threat {history['threat'][i+1]} | "
            f"interceptor {history['interceptor'][i+1]} | "
            f"reward {step['reward']}"
            + (outcome if outcome else "")
        )

        plt.pause(STEP_DELAY)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    env = AirDefenseEnv(10, 10, 10)

    model_path = os.path.join(os.path.dirname(__file__), "air_defense_ppo")
    if not os.path.exists(model_path + ".zip"):
        print("No saved model found — using random policy.")
        model = PPO("MlpPolicy", env, verbose=0)
    else:
        print(f"Loading model from {model_path}.zip")
        model = PPO.load(model_path, env=env)

    history, steps = run_episode(model, env)

    print(f"\n--- Episode finished in {len(steps)} steps ---\n")
    animate(history, steps, (env.width, env.height, env.length))
