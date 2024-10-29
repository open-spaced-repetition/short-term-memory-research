import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq  # type: ignore
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.optimize import minimize  # type: ignore
from itertools import accumulate
from fsrs_optimizer import power_forgetting_curve  # type: ignore

plt.style.use("ggplot")

plot = False
short_term_stabilty_list: list = []
dataset_path = "../anki-revlogs/revlogs"


def cum_concat(x):
    return list(accumulate(x))


def filter_revlog(entries):
    return filter(
        lambda entry: entry.button_chosen >= 1
        and (entry.review_kind != 3 or entry.ease_factor != 0),
        entries,
    )


def convert_native(entries):
    return map(
        lambda entry: {
            "review_time": entry.id,
            "card_id": entry.cid,
            "rating": entry.button_chosen,
            "review_state": entry.review_kind,
        },
        filter_revlog(entries),
    )


def format_t(s):
    return (
        f"{s:.2f}s"
        if s < 60
        else (
            f"{s/60:.2f}min"
            if s < 60 * 60
            else (
                f"{s/(60 * 60):.2f}h"
                if s < (60 * 60 * 24)
                else f"{s/(60 * 60 * 24):.2f}d"
            )
        )
    )


def fit_stability(delta_t, retention, size):
    def loss(stability):
        y_pred = power_forgetting_curve(delta_t, stability)
        loss = sum(
            -(retention * np.log(y_pred) + (1 - retention) * np.log(1 - y_pred)) * size
        )
        return loss

    res = minimize(loss, x0=1, bounds=[(0.1, None)])
    return res.x[0]


def to_days(value, position):
    return f"{value/60/60/24:.1f}"


def to_hours(value, position):
    return f"{value/60/60:.1f}"


def to_minutes(value, position):
    return f"{value/60:.2f}"


def process(user_id):
    df = pd.read_parquet(dataset_path, filters=[("user_id", "=", user_id)])

    if df.empty:
        return 0

    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["delta_t_f"] = df["elapsed_seconds"].map(format_t)
    df["t_bin"] = df["elapsed_seconds"].map(
        lambda x: (
            round(np.power(1.4, np.floor(np.log(x) / np.log(1.4))), 2) if x > 0 else 1
        )
    )
    t_history = df.groupby("card_id", group_keys=False)["elapsed_seconds"].apply(
        lambda x: cum_concat([[round(i, 2)] for i in x])
    )
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    r_history = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    t_f_history = df.groupby("card_id", group_keys=False)["delta_t_f"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["t_f_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_f_history for item in sublist
    ]
    df["y"] = df["rating"].map(lambda x: 1 if x > 1 else 0)
    df.to_csv(f"./processed/{user_id}.csv", index=False)

    for r_history in (
        "1",
        "1,2",
        "1,3",
        "1,3,3",
        "1,3,3,3",
        "2",
        "2,2",
        "2,3",
        "2,3,3",
        "3",
        "3,2",
        "3,3",
        "3,3,3",
    ):
        t_lim = df[df["r_history"] == r_history]["t_bin"].quantile(0.8)

        tmp = (
            df[(df["r_history"] == r_history) & (df["t_bin"] <= t_lim)]
            .groupby("t_bin")
            .agg({"y": ["mean", "count"]})
            .reset_index()
            .copy()
        )
        if tmp.empty:
            continue

        delta_t = tmp["t_bin"]
        y_mean = tmp["y"]["mean"]
        y_count = tmp["y"]["count"]
        sample_size = sum(y_count)
        if sample_size < 10:
            continue
        count_percent = np.array([x / sum(y_count) for x in y_count])
        s = max(round(fit_stability(delta_t, y_mean, y_count)), 1)
        s_text = format_t(s)
        average_delta_t = round(
            df[(df["r_history"] == r_history) & (df["t_bin"] <= t_lim)][
                "elapsed_seconds"
            ].mean()
        )
        average_delta_t_text = format_t(average_delta_t)
        average_retention = round(
            df[(df["r_history"] == r_history) & (df["t_bin"] <= t_lim)]["y"].mean(), 4
        )
        short_term_stabilty_list.append(
            (
                user_id,
                r_history,
                s,
                s_text,
                average_delta_t,
                average_delta_t_text,
                average_retention,
                sample_size,
            )
        )
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(delta_t, y_mean, s=count_percent * 1000, alpha=0.5)
            ax.plot(
                np.linspace(0, t_lim, 100),
                power_forgetting_curve(np.linspace(0, t_lim, 100), s),
                "r-",
                label=f"fit: s={s_text}",
            )
            ax.set_xlim(0, t_lim)
            if t_lim > 60 * 60 * 24 * 4:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(to_days))
                ax.set_xlabel("time (days)")
            elif t_lim > 60 * 60 * 4:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(to_hours))
                ax.set_xlabel("time (hours)")
            else:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(to_minutes))
                ax.set_xlabel("time (minutes)")
            ax.set_title(f"r_history={r_history} | sample_size={sample_size}")
            ax.set_ylim(None, 1)
            ax.set_ylabel("recall probability")
            ax.legend()
            fig.savefig(f"./short_term_forgetting_curve/{user_id}_{r_history}.png")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    plot = args.plot
    dataset = pq.ParquetDataset(dataset_path)
    users = sorted(dataset.partitioning.dictionaries[0], key=lambda x: x.as_py())[:64]
    Path("./short_term_forgetting_curve").mkdir(parents=True, exist_ok=True)
    for user in users:
        process(user)

    short_term_stabilty_df = pd.DataFrame(
        short_term_stabilty_list,
        columns=[
            "user",
            "r_history",
            "stability",
            "s_text",
            "average_delta_t",
            "average_delta_t_text",
            "average_retention",
            "sample_size",
        ],
    )
    short_term_stabilty_df.to_csv("./short_term_stability.tsv", sep="\t", index=False)
