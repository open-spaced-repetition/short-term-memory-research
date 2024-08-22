import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from stats_pb2 import RevlogEntries
from itertools import accumulate
from fsrs_optimizer import power_forgetting_curve

plt.style.use("ggplot")


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


def fit_stability(delta_t, retention, size):
    def loss(stability):
        y_pred = power_forgetting_curve(delta_t, stability)
        loss = sum(
            -(retention * np.log(y_pred) + (1 - retention) * np.log(1 - y_pred)) * size
        )
        return loss

    res = minimize(loss, x0=1)
    return res.x[0]


def to_days(value, position):
    return f"{value/1440:.1f}"


def to_hours(value, position):
    return f"{value/60:.2f}"


def process_revlog(revlog):
    data = open(revlog, "rb").read()
    entries = RevlogEntries.FromString(data)
    df = pd.DataFrame(convert_native(entries.entries))

    if df.empty:
        return 0

    df["is_learn_start"] = (df["review_state"] == 0) & (df["review_state"].shift() != 0)
    df["sequence_group"] = df["is_learn_start"].cumsum()
    last_learn_start = (
        df[df["is_learn_start"]].groupby("card_id")["sequence_group"].last()
    )
    df["last_learn_start"] = df["card_id"].map(last_learn_start).fillna(0).astype(int)
    df["mask"] = df["last_learn_start"] <= df["sequence_group"]
    df = df[df["mask"] == True]
    df = df.groupby("card_id").filter(lambda group: group["review_state"].iloc[0] == 0)

    df["review_time"] = df["review_time"].astype(int)
    df["delta_t"] = df["review_time"].diff().fillna(0).astype(int) // 1000
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.loc[df["i"] == 1, "delta_t"] = -1

    t_history = df.groupby("card_id", group_keys=False)["delta_t"].apply(
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

    for r_history in (
        "1",
        "1,3",
        "1,3,3",
        "1,3,3,3",
        "2",
        "2,3",
        "2,3,3",
        "3",
        "3,3",
        "3,3,3",
    ):
        df["t_bin"] = df["delta_t"].map(
            lambda x: (
                round(np.power(1.4, np.floor(np.log(x) / np.log(1.4))), 2)
                if x > 0
                else 0
            )
        )
        df["y"] = df["rating"].map(lambda x: 1 if x > 1 else 0)
        tmp = (
            df[df["r_history"] == r_history]
            .groupby("t_bin")
            .agg({"y": ["mean", "count"]})
            .reset_index()
            .copy()
        )
        tmp = tmp[(tmp["y"]["count"] >= 25) & (tmp["t_bin"] <= 1440)]
        if tmp.empty:
            continue

        delta_t = tmp["t_bin"]
        y_mean = tmp["y"]["mean"]
        y_count = tmp["y"]["count"]
        count_percent = np.array([x / sum(y_count) for x in y_count])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(delta_t, y_mean, s=count_percent * 1000, alpha=0.5)
        s = fit_stability(delta_t, y_mean, y_count)
        s_text = (
            f"{s:.2f}min"
            if s < 60
            else f"{s/60:.2f}h" if s < 1440 else f"{s/1440:.2f}d"
        )
        ax.plot(
            np.linspace(0, tmp["t_bin"].max(), 100),
            power_forgetting_curve(np.linspace(0, tmp["t_bin"].max(), 100), s),
            "r-",
            label=f"fit: s={s_text}",
        )
        ax.set_xlim(left=0)
        if s > 1440:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(to_days))
            ax.set_xlabel("time (days)")
        elif s > 60:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(to_hours))
            ax.set_xlabel("time (hours)")
        else:
            ax.set_xlabel("time (minutes)")
        ax.set_title(f"r_history={r_history}")
        ax.set_ylim(0.25, 1)
        ax.set_ylabel("recall probability")
        ax.legend()
        fig.savefig(f"./short_term_forgetting_curve/{revlog.stem}_{r_history}.png")
        plt.close(fig)


if __name__ == "__main__":
    dataset_path = "../FSRS-Anki-20k/revlogs/1/"
    files = sorted(Path(dataset_path).glob("*.revlog"), key=lambda x: int(x.stem))[:64]
    Path("./short_term_forgetting_curve").mkdir(parents=True, exist_ok=True)
    for path in files:
        process_revlog(path)
