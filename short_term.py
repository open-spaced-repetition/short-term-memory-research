import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq  # type: ignore
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.optimize import minimize  # type: ignore
from itertools import accumulate
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # type: ignore
import json
from fsrs_optimizer import power_forgetting_curve  # type: ignore

plt.style.use("ggplot")

plot = False
save_csv = False
dataset_path = "../anki-revlogs-10k/revlogs"
cards_path = "../anki-revlogs-10k/cards"
decks_path = "../anki-revlogs-10k/decks"
output_jsonl = "./short_term_stability.jsonl"


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


def fit_stability(delta_t, y):
    """Fit stability using raw timestamps and binary outcomes.

    Args:
        delta_t: 1D array-like of elapsed seconds since last review for each sample.
        y: 1D array-like of binary recall outcomes (1=remembered, 0=forgotten).

    Returns:
        Estimated stability scalar.
    """

    delta_t_arr = np.asarray(delta_t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    def loss(stability):
        y_pred = power_forgetting_curve(delta_t_arr, stability)
        # Numerical stability: clip probabilities away from 0 and 1
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        nll = -np.sum(y_arr * np.log(y_pred) + (1 - y_arr) * np.log(1 - y_pred))
        return nll

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
        return

    # Join cards and decks to obtain preset grouping
    try:
        cards_df = pd.read_parquet(cards_path, filters=[("user_id", "=", user_id)])
        if not cards_df.empty and "card_id" in cards_df.columns:
            df = df.merge(
                cards_df,
                on="card_id",
                how="left",
                suffixes=("", "_card"),
            )
    except Exception:
        pass

    try:
        decks_df = pd.read_parquet(decks_path, filters=[("user_id", "=", user_id)])
        if not decks_df.empty and "deck_id" in decks_df.columns:
            df = df.merge(
                decks_df,
                on="deck_id",
                how="left",
                suffixes=("", "_deck"),
            )
    except Exception:
        pass

    # Derive preset grouping column
    if "preset_id" in df.columns:
        df["preset_group"] = pd.to_numeric(df["preset_id"], errors="coerce").astype(
            "Int64"
        )
    else:
        df["preset_group"] = pd.Series([pd.NA] * len(df), dtype="Int64")

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
    if save_csv:
        df.to_csv(f"./processed/{user_id}.csv", index=False)

    # Iterate per preset group then per r_history
    out_lines: list[str] = []
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
        # For each preset group for this user
        for preset_value in sorted(df["preset_group"].dropna().unique()):
            group_df = df[(df["preset_group"] == preset_value)]
            t_lim = (
                group_df[group_df["r_history"] == r_history]["t_bin"].quantile(0.8)
                if plot
                else np.inf
            )

            subset = group_df[
                (group_df["r_history"] == r_history) & (group_df["t_bin"] <= t_lim)
            ]
            if subset.empty:
                continue
            sample_size = int(subset.shape[0])
            if sample_size < 10:
                continue

            s = max(
                round(
                    fit_stability(subset["elapsed_seconds"].values, subset["y"].values)
                ),
                1,
            )
            s_text = format_t(s)
            average_delta_t = round(subset["elapsed_seconds"].mean())
            median_delta_t = round(subset["elapsed_seconds"].median())
            average_delta_t_text = format_t(average_delta_t)
            average_retention = round(subset["y"].mean(), 4)

            # Buffer result to write once at the end for this user
            result = {
                "user": int(user_id),
                "preset": int(preset_value),
                "r_history": r_history,
                "stability": int(s),
                "s_text": s_text,
                "average_delta_t": int(average_delta_t),
                "average_delta_t_text": average_delta_t_text,
                "median_delta_t": int(median_delta_t),
                "average_retention": float(average_retention),
                "sample_size": int(sample_size),
            }
            out_lines.append(json.dumps(result) + "\n")
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                tmp = (
                    subset.groupby("t_bin")
                    .agg({"y": ["mean", "count"]})
                    .reset_index()
                    .copy()
                )
                delta_t = tmp["t_bin"]
                y_mean = tmp["y"]["mean"]
                y_count = tmp["y"]["count"]
                count_percent = np.array([x / max(1, sum(y_count)) for x in y_count])
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
                ax.set_title(
                    f"preset={preset_value} | r_history={r_history} | sample_size={sample_size}"
                )
                ax.set_ylim(None, 1)
                ax.set_ylabel("recall probability")
                ax.legend()
                fig.savefig(
                    f"./short_term_forgetting_curve/{user_id}_{preset_value}_{r_history}.png"
                )
                plt.close(fig)

    # Write this user's all results once
    if len(out_lines) > 0:
        with open(output_jsonl, "a") as f:
            f.writelines(out_lines)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save individual user CSV files to ./processed/",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh: delete existing JSONL and recompute all users",
    )
    args = parser.parse_args()
    plot = args.plot
    save_csv = args.save_csv
    dataset = pq.ParquetDataset(dataset_path)
    users = sorted(dataset.partitioning.dictionaries[0], key=lambda x: x.as_py())

    if plot:
        Path("./short_term_forgetting_curve").mkdir(parents=True, exist_ok=True)
    if save_csv:
        Path("./processed").mkdir(parents=True, exist_ok=True)

    # Build processed users set by scanning existing JSONL for any lines with 'user'
    processed_users = set()
    if Path(output_jsonl).exists() and not args.fresh:
        print(f"Resuming from existing {output_jsonl} (processed users only) ...")
        with open(output_jsonl, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "user" in obj:
                        processed_users.add(int(obj["user"]))
                except Exception:
                    continue

    # Optionally start fresh by removing the JSONL
    if Path(output_jsonl).exists() and args.fresh:
        Path(output_jsonl).unlink()
        print(f"Removed existing {output_jsonl}")

    # Users to process: skip already processed
    users_to_process = [u for u in users if int(u.as_py()) not in processed_users]

    # Use multiprocessing pool for parallel processing
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    print(
        f"Processing {len(users_to_process)} users (skipped {len(processed_users)}) with {n_jobs} parallel jobs..."
    )

    if len(users_to_process) > 0:
        with Pool(n_jobs) as pool:
            list(
                tqdm(
                    pool.imap(process, users_to_process),
                    total=len(users_to_process),
                    desc="Processing users",
                )
            )

    # Read jsonl file and convert to DataFrame
    print(f"Reading results from {output_jsonl}...")
    records = []
    with open(output_jsonl, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # No status-only lines to skip
            records.append(obj)

    short_term_stabilty_df = pd.DataFrame(records)
    # Ensure correct dtype for user column
    short_term_stabilty_df["user"] = (
        pd.to_numeric(short_term_stabilty_df["user"], errors="coerce")
        .astype("Int64")
        .astype("int64")
    )

    # Reorder columns to match original order
    # Include preset in final output if present
    desired_cols = [
        "user",
        "preset",
        "r_history",
        "stability",
        "s_text",
        "average_delta_t",
        "average_delta_t_text",
        "median_delta_t",
        "average_retention",
        "sample_size",
    ]
    existing_cols = [c for c in desired_cols if c in short_term_stabilty_df.columns]
    short_term_stabilty_df = short_term_stabilty_df[existing_cols]

    # Drop duplicates if any (safety): keep last by keys
    dup_keys = ["user", "r_history"] + (
        ["preset"] if "preset" in short_term_stabilty_df.columns else []
    )
    if len(dup_keys) > 0:
        short_term_stabilty_df = short_term_stabilty_df.drop_duplicates(
            subset=dup_keys, keep="last"
        )

    # Ensure correct dtype for preset column if present
    if "preset" in short_term_stabilty_df.columns:
        short_term_stabilty_df["preset"] = (
            pd.to_numeric(short_term_stabilty_df["preset"], errors="coerce")
            .astype("Int64")
            .astype("int64")
        )

    # Sort by user_id then preset if exists
    sort_keys = ["user"] + (
        ["preset"] if "preset" in short_term_stabilty_df.columns else []
    )
    short_term_stabilty_df = short_term_stabilty_df.sort_values(sort_keys).reset_index(
        drop=True
    )

    short_term_stabilty_df.to_csv("./short_term_stability.tsv", sep="\t", index=False)
    print(
        f"Done! Processed {len(records)} records and saved to short_term_stability.tsv"
    )
