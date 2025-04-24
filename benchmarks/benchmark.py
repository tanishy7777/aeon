import time
from timeit import Timer
from typing import List

import numpy as np
import pandas as pd

from benchmarks.sbd_distances import (
    sbd_distance,
    sbd_distance_fix,
    sbd_distance_pr2715
    # sbd_original,
    # sbd_tslearn,
)


def _timeit(func, repeat: int = 3) -> float:
    timer = Timer(func)
    number, time_taken = timer.autorange()
    raw_timings = timer.repeat(repeat=repeat - 1, number=number)
    raw_timings += [time_taken]
    timings = [dt / number for dt in raw_timings]
    best = min(timings)
    print(f"{number} loops, best of {repeat}: {best:.6f}s per loop")
    return best


def main():
    # TODO: becnhmark
    # - run it 2-3 times first
    # - then, benchmark with 1000, 5000, up to 25000 of length 100
    rng = np.random.default_rng(42)
    distance_funcs = [
        sbd_distance,
        sbd_distance_fix,
        sbd_distance_pr2715
        # sbd_original,
        # sbd_tslearn,
    ]
    timepoints_options = [10, 100, 1000, 10000, 25000]
    channel_options = [1, 10, 100, 1000, 5000]

    # warmup for Numba JIT
    print("Warming up Numba JIT...")
    ts1 = rng.random(100)
    ts2 = rng.random(100)
    for i in [1, 2, 5, 10]:
        for func in distance_funcs:
            # univariate
            func(ts1.reshape(1, -1), ts2.reshape(1, -1))
            # multivariate
            func(ts1.reshape(i, -1), ts2.reshape(i, -1))

    time.sleep(2)
    print("...done.")

    print("Starting benchmark (univariate)...")
    results = []
    for func in distance_funcs:
        print(f"  {func.__name__}:")
        for n_timepoints in timepoints_options:
            n_channels = 2
            ts1 = rng.random((n_channels, n_timepoints))
            ts2 = rng.random((n_channels, n_timepoints))
            print(f"    input=({n_channels}, {n_timepoints}): ", end="")
            # if func == sbd_tslearn or func == sbd_original:
            #     print("tslearn or original", func)
            #     ts1 = ts1.T
            #     ts2 = ts2.T
            #     best = _timeit(lambda: func(ts1, ts2))
            # else:
            print("aeon buggy/fixed", func)
            best = _timeit(lambda: func(ts1, ts2, standardize=False))

            results.append(
                {
                    "distance": func.__name__,
                    "n_channels": n_channels,
                    "n_timepoints_min": n_timepoints,
                    "n_timepoints_max": n_timepoints,
                    "type": "2",
                    "time": best,
                }
            )

        pd.DataFrame(results).to_csv("numba-benchmark.bak.csv", index=False)

        for n_timepoints in timepoints_options:
            n_channels = 1
            ts1 = rng.random((n_channels, n_timepoints))
            ts2 = rng.random((n_channels, n_timepoints))
            print(f"    input=({n_channels}, {n_timepoints}): ", end="")
            best = _timeit(lambda: func(ts1, ts2))
            results.append(
                {
                    "distance": func.__name__,
                    "n_channels": n_channels,
                    "n_timepoints_min": n_timepoints,
                    "n_timepoints_max": n_timepoints,
                    "type": "1",
                    "time": best,
                }
            )

        pd.DataFrame(results).to_csv("numba-benchmark.bak.csv", index=False)

        for n_channels in channel_options:
            n_timepoints = 1000
            ts1 = rng.random((n_channels, n_timepoints))
            ts2 = rng.random((n_channels, n_timepoints))
            print(f"    input=({n_channels}, {n_timepoints}): ", end="")
            best = _timeit(lambda: func(ts1, ts2))
            results.append(
                {
                    "distance": func.__name__,
                    "n_channels": n_channels,
                    "n_timepoints_min": n_timepoints,
                    "n_timepoints_max": n_timepoints,
                    "type": "3",
                    "time": best,
                }
            )
        pd.DataFrame(results).to_csv("numba-benchmark.bak.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("numba-benchmark.csv", index=False)
    print("...done.")


# n_channels = 2
# rng = np.random.default_rng(42)
# ts1 = rng.random((n_channels, 100))
# ts2 = rng.random((n_channels, 100))

# print(ts1.shape, ts2.shape)
# print(sbd_original(ts1.T, ts2.T))
# print(sbd_tslearn(ts1.T, ts2.T))
# print(sbd_distance(ts1, ts2, standardize=False))
# print(sbd_distance_fix(ts1, ts2, standardize=False))

if __name__ == "__main__":
    main()
