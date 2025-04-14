from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    results_path = Path("numba-benchmark.csv")
    if not results_path.exists():
        raise FileNotFoundError(
            "No results found, please either try with a backup or re-run the benchmark!"
        )

    df = pd.read_csv(results_path)
    # print("Loaded results from", df.shape, df.head())
    # print(df["distance"].unique())
    df["distance_func"] = df["distance"]
    # print(df)
    # mask = (df["n_channels"] == 2) & (df["type"] == 2)
    # print(df.loc[mask])
    # print()

    # print(df.loc[mask, "time"])
    # mask = (df["n_channels"] == 1) & (df["type"] == 1)
    # print(df.loc[mask])
    # print(df.loc[mask, "time"])
    # print()
    # mask = df["type"] == 3
    # print(df.loc[mask])
    # print(df.loc[mask, "time"])
    fig, axs = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")
    fig.suptitle(f"Runtime vs. Time Series Length channels: 1")
    axs.set_title("sbd")
    axs.set_xlabel("time series length")
    axs.set_ylabel("runtime (s)")
    for distance_func in df["distance_func"].unique():
        mask = (
            (df["distance_func"] == distance_func)
            & (df["n_channels"] == 1)
            & (df["type"] == 1)
        )
        if distance_func == "sbd_original":
            axs.plot(
                df.loc[mask, "n_timepoints_max"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.5,
            )
        elif distance_func == "sbd_distance_fix":
            axs.plot(
                df.loc[mask, "n_timepoints_max"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.5,
                color="red",
            )
        else:
            axs.plot(
                df.loc[mask, "n_timepoints_max"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.7,
            )

    axs.legend()

    fig, axs = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")
    fig.suptitle(f"Runtime vs. Time Series Length channels: 2")
    axs.set_title("sbd")
    axs.set_xlabel("time series length")
    axs.set_ylabel("runtime (s)")
    for distance_func in df["distance_func"].unique():
        print(distance_func)
        mask = (
            (df["distance_func"] == distance_func)
            & (df["n_channels"] == 2)
            & (df["type"] == 2)
        )
        if distance_func == "sbd_original":
            axs.plot(
                df.loc[mask, "n_timepoints_max"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.5,
            )
        elif distance_func == "sbd_distance_fix":
            axs.plot(
                df.loc[mask, "n_timepoints_max"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.5,
                color="red",
            )
        else:
            axs.plot(
                df.loc[mask, "n_timepoints_max"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.7,
            )
    axs.legend()

    fig, axs = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")
    fig.suptitle(f"Runtime vs. Time Series timepoints: 1000")
    axs.set_title("sbd")
    axs.set_xlabel("time series length")
    axs.set_ylabel("runtime (s)")
    for distance_func in df["distance_func"].unique():
        print(distance_func)
        mask = (
            (df["distance_func"] == distance_func)
            & (df["n_timepoints_min"] == 1000)
            & (df["type"] == 3)
        )
        if distance_func == "sbd_original":
            axs.plot(
                df.loc[mask, "n_channels"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.5,
            )
        elif distance_func == "sbd_distance_fix":
            axs.plot(
                df.loc[mask, "n_channels"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.5,
                color="red",
            )
        else:
            axs.plot(
                df.loc[mask, "n_channels"],
                df.loc[mask, "time"],
                label=distance_func,
                alpha=0.7,
            )
    axs.legend()

    # fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    # fig.suptitle("Runtime vs. Time Series Length channels: 1")
    # for i, distance in enumerate(distances):
    #     df_tmp = df[df["distance"] == distance]
    #     axs[i].set_title(distance)
    #     axs[i].set_xlabel("time series length")
    #     axs[i].set_ylabel("runtime (s)")
    #     for distance_func in df_tmp["distance_func"].unique():
    #         mask = (df_tmp["distance_func"] == distance_func) & (df_tmp["n_channels"] == 1)
    #         axs[i].plot(df_tmp.loc[mask, "n_timepoints_max"], df_tmp.loc[mask, "time"], label=distance_func,
    #                     alpha=0.5)
    #     axs[i].legend()

    # fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    # fig.suptitle("Runtime vs. No of channels, timesteps: 1000")
    # for i, distance in enumerate(distances):
    #     df_tmp = df[df["distance"] == distance]
    #     axs[i].set_title(distance)
    #     axs[i].set_xlabel("time series length")
    #     axs[i].set_ylabel("runtime (s)")
    #     for distance_func in df_tmp["distance_func"].unique():
    #         mask = (df_tmp["distance_func"] == distance_func) & (df_tmp["n_timepoints_min"] == 1000)
    #         axs[i].plot(df_tmp.loc[mask, "n_channels"], df_tmp.loc[mask, "time"], label=distance_func,
    #                     alpha=0.5)
    #     axs[i].legend()

    plt.show()


if __name__ == "__main__":
    main()
