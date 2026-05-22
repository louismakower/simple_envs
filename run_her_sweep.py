"""HER parameter sweep launcher for the one_dim and two_dim SAC agents.

Runs, per environment, every combination of:
  * HER on  -- mode in {future, final} x k in {1, 2, 4}   (6 configs)
  * HER off -- a single `nohier` config
each repeated over 5 seeds (0-4) => 7 configs x 5 seeds x 2 envs = 70 runs.

Each run is an independent `python -m {env}.train` subprocess writing to
  runs/her_sweep/{sac_1d,sac_2d}/{config}_seed{s}/
A finished run drops a `DONE` file; re-running the launcher skips those, so it
can be interrupted and resumed.

Usage:
  python run_her_sweep.py --jobs 4              # run the full sweep, 4 in parallel
  python run_her_sweep.py --envs one_dim        # restrict to one env
  python run_her_sweep.py --dry-run             # just print the commands
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

ENVS = {"one_dim": "sac_1d", "two_dim": "sac_2d"}
MODES = ["future", "final"]
KS = [1, 2, 4]
SEEDS = list(range(5))
SWEEP_ROOT = os.path.join("runs", "her_sweep")


def build_configs():
    """Return list of (config_name, her_flags) covering the HER grid."""
    configs = [("nohier", ["--disable-her"])]
    for mode in MODES:
        for k in KS:
            configs.append((f"{mode}_k{k}", ["--her-mode", mode, "--her-k", str(k)]))
    return configs


def build_jobs(envs):
    """Return list of job dicts for the requested envs."""
    jobs = []
    for env in envs:
        exp = ENVS[env]
        log_dir = os.path.join(SWEEP_ROOT, exp)
        for config_name, her_flags in build_configs():
            for seed in SEEDS:
                run_name = f"{config_name}_seed{seed}"
                run_dir = os.path.join(ROOT, log_dir, run_name)
                cmd = [
                    sys.executable, "-m", f"{env}.train",
                    "--agent", "sac",
                    "--headless", "--no-visualise",
                    "--log-dir", log_dir,
                    "--run-name", run_name,
                    "--seed", str(seed),
                    *her_flags,
                ]
                jobs.append({"env": env, "run_name": run_name,
                             "run_dir": run_dir, "cmd": cmd})
    return jobs


def run_job(job):
    """Run one training subprocess; return (run_name, status)."""
    run_dir = job["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    print(f"[START] {job['env']}/{job['run_name']}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(job["cmd"], cwd=ROOT,
                                stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        print(f"[OK]    {job['env']}/{job['run_name']}")
        return job["run_name"], "ok"
    print(f"[FAIL]  {job['env']}/{job['run_name']} (exit {result.returncode}, see {log_path})")
    return job["run_name"], "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=4,
                        help="Number of runs to execute concurrently.")
    parser.add_argument("--envs", nargs="+", choices=list(ENVS), default=list(ENVS),
                        help="Which environments to sweep (default: both).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would run, then exit.")
    args = parser.parse_args()

    jobs = build_jobs(args.envs)

    if args.dry_run:
        for job in jobs:
            print(" ".join(job["cmd"]))
        print(f"\n{len(jobs)} runs total.")
        return

    pending, skipped = [], []
    for job in jobs:
        if os.path.exists(os.path.join(job["run_dir"], "DONE")):
            skipped.append(job["run_name"])
        else:
            pending.append(job)

    print(f"{len(jobs)} runs total | {len(skipped)} already done | "
          f"{len(pending)} to run | {args.jobs} parallel\n")

    results = {"ok": [], "failed": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for run_name, status in pool.map(run_job, pending):
            results[status].append(run_name)

    print("\n=== Sweep summary ===")
    print(f"  ok      : {len(results['ok'])}")
    print(f"  skipped : {len(skipped)}")
    print(f"  failed  : {len(results['failed'])}")
    for run_name in results["failed"]:
        print(f"    FAILED: {run_name}")
    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
