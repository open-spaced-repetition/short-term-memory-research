from itertools import accumulate
from sklearn.isotonic import IsotonicRegression  # type: ignore
import pandas as pd
import numpy as np
from scipy.optimize import minimize  # type: ignore
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # type: ignore

DECAY = -0.5


def power_forgetting_curve(t, s, decay=DECAY):
    factor = 0.9 ** (1 / decay) - 1
    return (1 + factor * t / s) ** decay


def fit_stability(delta_t, retention, size, decay=DECAY):

    def loss(stability):
        y_pred = power_forgetting_curve(delta_t, stability, decay).clip(
            1e-10, 1 - 1e-10
        )
        loss = sum(
            -(retention * np.log(y_pred) + (1 - retention) * np.log(1 - y_pred)) * size
        )
        return loss

    res = minimize(loss, x0=1, bounds=[(0.1, None)])
    return res.x[0]


def format_time(x, pos=None):
    if x < 60:
        return f"{x:.0f}s"
    elif x < 3600:
        return f"{x/60:.2f}min"
    elif x < 86400:
        return f"{x/3600:.2f}h"
    else:
        return f"{x/86400:.2f}d"


def filter_data(data):
    Q1 = data["elapsed_seconds"].quantile(0.25)
    Q3 = data["elapsed_seconds"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[
        (data["elapsed_seconds"] >= lower_bound)
        & (data["elapsed_seconds"] <= upper_bound)
    ]
    return data


def process(user_id):
    df = pd.read_parquet(
        "../anki-revlogs-10k/revlogs", filters=[("user_id", "=", user_id)]
    )
    df["review_th"] = range(1, len(df) + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)

    def cum_concat(x):
        return list(accumulate(x))

    t_history_list = df.groupby("card_id", group_keys=False)["elapsed_seconds"].apply(
        lambda x: cum_concat([[max(0, i)] for i in x])
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
    df["y"] = df["rating"].map(lambda x: 1 if x > 1 else 0)
    df.head()

    r_history_list = ["1,1", "1,2", "1,3", "2,1", "2,2", "2,3", "3,1", "3,2", "3,3", "1,3,3"]
    r_history_delta_t_bounds = {}
    N = 5  # Can be adjusted to any number

    df = df[df["r_history"].isin(r_history_list + ["1", "2", "3"])]
    df = (
        df.groupby("r_history", as_index=True)
        .apply(filter_data, include_groups=False)
        .reset_index(level="r_history")
    )

    stats_data = {}

    for r_history in r_history_list:

        stats_data[r_history] = []

        data = df[df["r_history"] == r_history][["elapsed_seconds", "y", "t_history"]]
        # data = filter_data(data)
        r_history_delta_t_bounds[r_history] = (
            data["elapsed_seconds"].min(),
            data["elapsed_seconds"].max(),
        )

        if len(r_history) > 1:
            prev_data = df[df["r_history"] == r_history[:-2]]
            if prev_data.empty:
                continue
            # prev_data = filter_data(prev_data)
            prev_stability = fit_stability(
                prev_data["elapsed_seconds"],
                prev_data["y"],
                np.ones_like(prev_data["elapsed_seconds"]),
            )

            # Get previous interval from t_history
            data["prev_interval"] = data["t_history"].apply(
                lambda x: [float(i) for i in x.split(",")][-1]
            )

            # Split into N groups based on percentiles
            percentiles = np.linspace(0, 100, N + 1)
            interval_bounds = np.percentile(prev_data["elapsed_seconds"], percentiles)

            datasets = []

            for i in range(N):
                # Filter data for current interval range
                mask = (data["prev_interval"] > interval_bounds[i]) & (
                    data["prev_interval"] <= interval_bounds[i + 1]
                )
                if mask.sum() == 0:
                    continue
                
                datasets.append(data[mask])

                # Calculate stats for current interval range
                prev_mask = (prev_data["elapsed_seconds"] > interval_bounds[i]) & (
                    prev_data["elapsed_seconds"] <= interval_bounds[i + 1]
                )
                # median_ivl = prev_data[prev_mask]["elapsed_seconds"].median()
                avg_ivl = prev_data[prev_mask]["elapsed_seconds"].mean()
                avg_retention = prev_data[prev_mask]["y"].mean()

                stats_data[r_history].append(
                    {
                        "r_history": r_history,
                        "prev_stability": prev_stability,
                        "prev_avg_interval": avg_ivl,
                        "prev_avg_retention": avg_retention,
                    }
                )
        else:
            datasets = [data]

        for idx, dataset in enumerate(datasets):
            x, y = dataset.sort_values(by="elapsed_seconds")[
                ["elapsed_seconds", "y"]
            ].values.T

            ir = IsotonicRegression(y_min=0, y_max=1, increasing=False)
            ir.fit(x, y)
            y_ir = ir.predict(x)

            s_seconds = fit_stability(x, y_ir, np.ones_like(x))
            cnt = len(x)

            stats_data[r_history][idx]["stability"] = s_seconds
            stats_data[r_history][idx]["sample_size"] = cnt

    flat_data = []
    for r_history, groups in stats_data.items():
        for group in groups:
            flat_data.append(group)

    df_stats = pd.DataFrame(flat_data)

    df_stats = df_stats[
        [
            "r_history",
            "prev_stability",
            "prev_avg_interval",
            "prev_avg_retention",
            "stability",
            "sample_size",
        ]
    ]

    df_stats["SInc"] = df_stats["stability"] / df_stats["prev_stability"]
    df_stats["user_id"] = user_id

    df_stats.to_csv(f"./short_term_stats/{user_id}.csv", index=False)


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process, range(1, 65)), total=64))
