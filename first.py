from fsrs_optimizer import (  # type: ignore
    power_forgetting_curve,
    fit_stability,
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyarrow.parquet as pq  # type: ignore
from pathlib import Path
from itertools import accumulate

plt.style.use("ggplot")
dataset_path = "../anki-revlogs-10k/revlogs"


def cum_concat(x):
    return list(accumulate(x))


def create_time_series(df):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df = df[df["rating"].isin([1, 2, 3, 4])].copy()
    df["i"] = df.groupby("card_id").cumcount() + 1
    t_history_list = df.groupby("card_id", group_keys=False)["elapsed_days"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history_list for item in sublist
    ]
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df = df[df["elapsed_days"] != 0].copy()
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["first_rating"] = df["r_history"].map(lambda x: x[0] if len(x) > 0 else "")
    df.dropna(inplace=True)
    return df[df["elapsed_days"] > 0].sort_values(by=["review_th"])


if __name__ == "__main__":
    dataset = pq.ParquetDataset(dataset_path)
    plot_path = Path("./first_forgetting_curve")
    plot_path.mkdir(exist_ok=True)
    users = sorted(dataset.partitioning.dictionaries[0], key=lambda x: x.as_py())[:64]
    for user_id in users:
        df = create_time_series(
            pd.read_parquet(dataset_path, filters=[("user_id", "=", user_id)])
        )

        ratings = set()
        for r_history in df[df["i"] == 2]["r_history"].value_counts().head(5).index:
            tmp = (
                df[df["r_history"] == r_history]
                .groupby("elapsed_days")
                .agg(
                    {
                        "y": ["mean", "count"],
                    }
                )
            )
            delta_t = tmp.index
            y_mean = tmp["y"]["mean"]
            y_count = tmp["y"]["count"]
            count_percent = np.array([x / sum(y_count) for x in y_count])
            stability = fit_stability(delta_t, y_mean, y_count)

            plt.figure(r_history[0])
            ratings.add(r_history[0])
            plt.plot(
                np.linspace(0, 10),
                power_forgetting_curve(np.linspace(0, 10), stability),
                label=f"r_history={r_history}|stability={stability:.2f}|count={sum(y_count)}",
            )
            plt.scatter(delta_t, y_mean, count_percent * 1000, alpha=0.5)
            plt.xlim(0, 10)
            plt.ylim(0, 1)
            plt.legend()
            plt.xlabel("delta_t")
            plt.ylabel("retention")
            plt.title(f"user {user_id}")

        for rating in ratings:
            plt.figure(str(rating))
            plt.savefig(plot_path / f"{user_id}-{rating}.png")
            plt.cla()
