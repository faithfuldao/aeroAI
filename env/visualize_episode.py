import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import SAC
from air_defense_env import AirDefenseEnv, KILL_RADIUS, ZONE_XY, ZONE_Z

STEP_DELAY    = 0.2
EPISODE_PAUSE = 1.0
NUM_EPISODES  = 0  


def run_episode(model, env):
    obs, _ = env.reset()
    done = False
    history = {
        "missile":      [env.missile.position.copy()],
        "interceptor":  [env.interceptor.position.copy()],
        "target":       env.target_pos.copy(),
    }
    steps = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        history["missile"].append(env.missile.position.copy())
        history["interceptor"].append(env.interceptor.position.copy())
        steps.append({"reward": reward, "done": done or truncated})
        if truncated:
            break

    return history, steps


def animate_episode(ax, history, steps, episode_num, stats):
    total_steps = len(steps)
    last_dist = np.linalg.norm(
        history["missile"][-1] - history["interceptor"][-1]
    )
    intercepted = last_dist < KILL_RADIUS
    outcome_label = "INTERCEPTED" if intercepted else "TARGET HIT"

    missile_trail_x, missile_trail_y, missile_trail_z = [], [], []
    inter_trail_x,   inter_trail_y,   inter_trail_z   = [], [], []

    print(f"\n--- Episode {episode_num} ({total_steps} steps) ---")

    for i, step in enumerate(steps):
        if not plt.fignum_exists(ax.figure.number):
            return False

        ax.cla()

        m = history["missile"][i + 1]
        it = history["interceptor"][i + 1]
        missile_trail_x.append(m[0]);  missile_trail_y.append(m[1]);  missile_trail_z.append(m[2])
        inter_trail_x.append(it[0]);   inter_trail_y.append(it[1]);   inter_trail_z.append(it[2])

        ax.plot(missile_trail_x, missile_trail_y, missile_trail_z, "r--", alpha=0.35, linewidth=1)
        ax.plot(inter_trail_x,   inter_trail_y,   inter_trail_z,   "g--", alpha=0.35, linewidth=1)

        ax.scatter(*history["target"],       color="blue",  s=120, zorder=5, label="Target")
        ax.scatter(*m,                        color="red",   s=120, zorder=5, label="Missile")
        ax.scatter(*it,                       color="green", s=120, zorder=5, label="Interceptor")
        ax.scatter(*history["missile"][0],    color="red",   s=40,  alpha=0.2)
        ax.scatter(*history["interceptor"][0],color="green", s=40,  alpha=0.2)

        dist = np.linalg.norm(m - it)
        outcome = f"  *** {outcome_label} ***" if step["done"] else ""
        ax.set_title(
            f"Episode {episode_num} | Step {i+1}/{total_steps} | "
            f"dist {dist:.0f} m | reward {step['reward']:.1f}{outcome}\n"
            f"wins: {stats['wins']}  losses: {stats['losses']}"
        )
        ax.set_xlim(0, ZONE_XY); ax.set_ylim(0, ZONE_XY); ax.set_zlim(0, ZONE_Z)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Altitude (m)")
        ax.legend(loc="upper left")

        print(f"  step {i+1:>3} | missile {m.astype(int)} | interceptor {it.astype(int)} | dist {dist:.0f} m{outcome}")

        plt.pause(EPISODE_PAUSE if step["done"] else STEP_DELAY)

    return True


if __name__ == "__main__":
    env = AirDefenseEnv()

    model_path = os.path.join(os.path.dirname(__file__), "air_defense_sac")
    if not os.path.exists(model_path + ".zip"):
        print("No saved model found — using untrained policy.")
        model = SAC("MlpPolicy", env, verbose=0)
    else:
        print(f"Loading model from {model_path}.zip")
        model = SAC.load(model_path, env=env)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()

    stats = {"wins": 0, "losses": 0}
    episode = 0

    while NUM_EPISODES == 0 or episode < NUM_EPISODES:
        episode += 1
        history, steps = run_episode(model, env)

        last_dist = np.linalg.norm(history["missile"][-1] - history["interceptor"][-1])
        if last_dist < KILL_RADIUS:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        still_open = animate_episode(ax, history, steps, episode, stats)
        if not still_open:
            break

    print(f"\n=== Done | {stats['wins']} wins / {stats['losses']} losses ===")
    plt.ioff()
    plt.show()
