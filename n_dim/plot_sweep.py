import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from tensorboard.compat.proto import event_pb2

LOG_DIR = "runs/sweep"
from n_dim.sweep import log_path


def _cache_path(run_path, tag):
    safe_tag = tag.replace("/", "_")
    return os.path.join(run_path, f".cache_{safe_tag}.npz")


def _read_events_fast(filepath, tag, max_record_bytes=500):
    steps, values = [], []
    with open(filepath, "rb") as f:
        while True:
            header = f.read(12)  # uint64 length + uint32 length_crc
            if len(header) < 12:
                break
            length = struct.unpack_from("<Q", header)[0]
            if length > max_record_bytes:
                f.seek(length + 4, 1)  # skip data + data_crc
                continue
            data = f.read(length)
            if len(data) < length:
                break
            f.read(4)  # skip data_crc
            try:
                ev = event_pb2.Event()
                ev.ParseFromString(data)
                if ev.HasField("summary"):
                    for v in ev.summary.value:
                        if v.tag == tag and v.HasField("simple_value"):
                            steps.append(ev.step)
                            values.append(v.simple_value)
            except Exception:
                continue
    return steps, values


def read_scalar(path, tag):
    cache = _cache_path(path, tag)
    event_files = [f for f in os.listdir(path) if f.startswith("events.out.tfevents")]
    event_mtime = max(os.path.getmtime(os.path.join(path, f)) for f in event_files) if event_files else 0

    if os.path.exists(cache) and os.path.getmtime(cache) >= event_mtime:
        try:
            data = np.load(cache)
            return data["steps"], data["values"]
        except Exception:
            # Truncated/corrupt cache (e.g. run interrupted mid-write); re-parse.
            pass

    steps, values = [], []
    for fname in sorted(event_files):
        s, v = _read_events_fast(os.path.join(path, fname), tag)
        steps.extend(s)
        values.extend(v)

    steps = np.array(steps)
    values = np.array(values)
    if len(steps):
        np.savez(cache, steps=steps, values=values)
    return steps, values


COLORS = {
    "PPO":     "tab:blue",
    "SAC+HER": "tab:green",
    "SAC":     "tab:orange",
}


def load_all(conditions, n_values, seeds):
    jobs = [(cond, n, seed) for cond in conditions for n in n_values for seed in seeds]

    def _load(job):
        cond, n, seed = job
        path = log_path(cond, n, seed)
        if not os.path.exists(path):
            return job, None
        return job, read_scalar(path, cond["reward_tag"])

    results = {}
    with ThreadPoolExecutor() as pool:
        for job, data in pool.map(_load, jobs):
            if data is not None:
                results[job[0]["label"], job[1], job[2]] = data
    return results


def plot_results():
    data = load_all(conditions, n_values, seeds)

    ncols = math.ceil(math.sqrt(len(n_values)))
    nrows = math.ceil(len(n_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, n in zip(axes_flat, n_values):
        for cond in conditions:
            seed_curves = []
            ref_steps = None

            for seed in seeds:
                entry = data.get((cond["label"], n, seed))
                if entry is None:
                    continue
                steps, values = entry
                if len(steps) == 0:
                    continue
                if ref_steps is None:
                    ref_steps = steps
                if len(steps) == len(ref_steps):
                    seed_curves.append(values)
                else:
                    seed_curves.append(np.interp(ref_steps, steps, values))

            if not seed_curves:
                continue

            curves = np.stack(seed_curves)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            color = COLORS[cond["label"]]

            ax.plot(ref_steps / 1e6, mean, color=color)
            ax.fill_between(ref_steps / 1e6, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_title(f"n={n}")
        ax.set_xlabel("Environment steps (M)")

    for ax in axes_flat[len(n_values):]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("Mean reward")

    legend_handles = [
        mlines.Line2D([], [], color=COLORS[cond["label"]], label=cond["label"])
        for cond in conditions
    ]
    fig.legend(handles=legend_handles, loc="center right", title="Algorithm")
    fig.suptitle(f"Reward vs dimensionality (±1 std, {len(seeds)} seeds)")
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    out_path = os.path.join(LOG_DIR, "reward_curves.png")
    os.makedirs(LOG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    seeds = [42, 43, 44]
    n_values = [1, 2, 3, 4,5, 6, 7,8, 9,10]
    conditions = [
        dict(label="PPO",     agent="ppo", extra_args=[],                  reward_tag="rewards/mean"),
        dict(label="SAC+HER", agent="sac", extra_args=[],                  reward_tag="rewards/extrinsic_mean"),
        dict(label="SAC",     agent="sac", extra_args=["--disable_her"],   reward_tag="rewards/extrinsic_mean"),
    ]
    plot_results()
