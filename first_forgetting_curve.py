import os
import sys

sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))
from fsrs_optimizer import remove_outliers, power_forgetting_curve

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

plt.style.use("ggplot")

for path in Path("../fsrs-dataset/short-term/").glob("revlog_history-*.tsv"):
    df = pd.read_csv(path, sep="\t")
    try:
        df[df["i"] == 2] = (
            df[df["i"] == 2]
            .groupby(by=["r_history", "t_history"], as_index=False, group_keys=False)
            .apply(remove_outliers)
        )
    except:
        continue
    
    ratings = set()
    for r_history in df[df["i"] == 2]["r_history"].value_counts().head(5).index:
        tmp = (
            df[df["r_history"] == r_history]
            .groupby("delta_t")
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
        weight = np.sqrt(y_count)

        def loss(stability):
            y_pred = power_forgetting_curve(delta_t, stability)
            logloss = sum(
                -(y_mean * np.log(y_pred) + (1 - y_mean) * np.log(1 - y_pred)) * weight
            )
            return logloss

        res = minimize(
            loss,
            x0=1,
            bounds=((0.01, 100),),
        )
        params = res.x
        stability = params[0]

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
        plt.title(f"user {path.stem.split('-')[1]}")

    for rating in ratings:
        plt.figure(str(rating))
        plt.savefig(f"./first_forgetting_curve/{path.stem.split('-')[1]}-{rating}.png")
        plt.cla()