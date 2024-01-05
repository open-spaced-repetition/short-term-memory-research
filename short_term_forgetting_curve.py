import pandas as pd
from itertools import accumulate
import math
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from fsrs_optimizer import Optimizer
from pathlib import Path
import hashlib


def cum_concat(x):
    return list(accumulate(x))


def power_forgetting_curve(t, s):
    return (1 + t / (9 * s)) ** -1


def to_minutes(seconds):
    return f"{seconds/60:.2f}"


def process(filename):
    sha256 = hashlib.sha256()
    sha256.update(filename.stem.encode("utf-8"))
    hash_id = sha256.hexdigest()[:7]
    optimizer = Optimizer()
    optimizer.anki_extract(filename=filename)
    df = pd.read_csv("./revlog.csv")
    df["delta_t"] = df["review_time"].diff().fillna(0) / 1000 / 60
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.loc[df["i"] == 1, "delta_t"] = 0

    t_history = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[round(i, 2)] for i in x])
    )
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    r_history = df.groupby("card_id", group_keys=False)["review_rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]

    for r_history in ("1", "1,3", "1,3,3", "1,3,3,3", "2", "2,3", "2,3,3", "3", "3,3", "3,3,3"):
        df["t_bin"] = df["delta_t"].map(
            lambda x: round(math.pow(1.4, math.floor(math.log(x, 1.4))), 2)
            if x > 0
            else 0
        )
        df["y"] = df["review_rating"].map(lambda x: 1 if x > 1 else 0)
        tmp = (
            df[df["r_history"] == r_history]
            .groupby("t_bin")
            .agg({"y": ["mean", "count"]})
            .reset_index()
            .copy()
        )
        tmp = tmp[tmp["y"]["count"] >= 25]
        if tmp.empty:
            continue
        plt.scatter(tmp["t_bin"], tmp[("y", "mean")], s=tmp[("y", "count")])
        popt, pcov = curve_fit(
            power_forgetting_curve,
            tmp["t_bin"],
            tmp[("y", "mean")],
            sigma=1 / tmp[("y", "count")],
        )
        stability = (
            f"{popt[0]:.2f}min"
            if popt[0] < 60
            else f"{popt[0]/60:.2f}h"
            if popt[0] < 1440
            else f"{popt[0]/1440:.2f}d"
        )
        plt.plot(
            tmp["t_bin"],
            power_forgetting_curve(tmp["t_bin"], *popt),
            "r-",
            label=f"fit: s={stability}",
        )
        plt.title(f"r_history={r_history}")
        plt.ylim(0.25, 1)
        plt.xlabel("delta_t (minutes)")
        plt.ylabel("recall probability")
        plt.legend()
        plt.grid()
        plt.savefig(f"./short_term_forgetting_curve/{hash_id}_{r_history}.png")
        plt.clf()


if __name__ == "__main__":
    dataset_path = "../fsrs-optimizer/dataset"
    files = Path(dataset_path).iterdir()
    Path("./plots").mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.suffix not in [".apkg", ".colpkg"]:
            continue
        process(file)
