"""
Build a centralized supervised-learning table from the scattered core (kernel) source files.

Row granularity: one row per (case, hour) -- 168 hours per case.
Columns: id/time columns, then raw joined features (price, inflow, storage, water value,
all per-reservoir where applicable), then target columns (one per unit) at the end.

No feature engineering here (no normalization, no lags, no derived columns) -- this is a
straight join of the raw core dataset into a flat table, as discussed. Extended data
(topology, constraints) is intentionally left out for now.

Output: data/processed/train_hourly.parquet (has targets) and
        data/processed/test_hourly.parquet (features only, to be predicted on).
"""
import re
import numpy as np
import pandas as pd

DATA = "data"
N_HOURS = 168


def sanitize(name: str) -> str:
    return name.replace(" ", "_")


def main():
    price = pd.read_csv(f"{DATA}/kernel/Historical_day_ahead_price_2015_2025.csv")
    inflow = pd.read_csv(f"{DATA}/kernel/Historical_inflow_1958_2025.csv")
    volume = pd.read_csv(f"{DATA}/kernel/Historical_volume_2015_2024.csv")
    water_value = pd.read_csv(f"{DATA}/kernel/Synthetic_water_value_2015_2024.csv")
    uc = pd.read_csv(f"{DATA}/kernel/Unit_commitment_decisions.csv")
    pred_mapping = pd.read_csv("prediction_mapping.csv")

    for df in (price, inflow, volume, water_value):
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    uc["starttime"] = pd.to_datetime(uc["starttime"])
    pred_mapping["starttime"] = pd.to_datetime(pred_mapping["starttime"])

    price = price.rename(columns={"date": "timestamp", "Day-ahead price (EUR/MWh)": "price_eur_mwh"})
    inflow = inflow.rename(columns={"date": "timestamp"})
    inflow = inflow.rename(columns={c: f"inflow_{sanitize(c)}" for c in inflow.columns if c != "timestamp"})
    volume = volume.rename(columns={"date": "day"})
    volume = volume.rename(columns={c: f"storage_{sanitize(c)}" for c in volume.columns if c != "day"})
    water_value = water_value.rename(columns={"date": "day"})
    water_value = water_value.rename(columns={c: f"water_value_{sanitize(c)}" for c in water_value.columns if c != "day"})

    # --- spine: one row per (case, hour) for both train and test cases ---
    cases = pd.concat([
        uc[["Run No", "starttime"]].assign(split="train"),
        pred_mapping[["Run No", "starttime"]].assign(split="test"),
    ], ignore_index=True).rename(columns={"Run No": "run_no"})

    n_cases = len(cases)
    run_no_rep = np.repeat(cases["run_no"].values, N_HOURS)
    starttime_rep = np.repeat(cases["starttime"].values, N_HOURS)
    split_rep = np.repeat(cases["split"].values, N_HOURS)
    hour_rep = np.tile(np.arange(N_HOURS), n_cases)
    timestamp = starttime_rep + hour_rep * np.timedelta64(1, "h")

    long_df = pd.DataFrame({
        "run_no": run_no_rep,
        "split": split_rep,
        "starttime": starttime_rep,
        "hour": hour_rep,
        "timestamp": timestamp,
    })
    long_df["day"] = long_df["timestamp"].dt.floor("D")

    # --- join raw features ---
    long_df = long_df.merge(price, on="timestamp", how="left")
    long_df = long_df.merge(inflow, on="timestamp", how="left")
    long_df = long_df.merge(volume, on="day", how="left")
    long_df = long_df.merge(water_value, on="day", how="left")

    missing = long_df.filter(regex="^(price_|inflow_|storage_|water_value_)").isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("WARNING - unmatched rows after join:")
        print(missing)

    # --- reshape targets (train only) from wide to long ---
    label_cols = [c for c in uc.columns if c.startswith("result_committed_")]
    units = sorted({re.match(r"result_committed_(.+)_t\d+$", c).group(1) for c in label_cols})

    target_frames = []
    uc_indexed = uc.set_index("Run No")
    for u in units:
        cols = [f"result_committed_{u}_t{t}" for t in range(N_HOURS)]
        sub = uc_indexed[cols].copy()
        sub.columns = pd.RangeIndex(N_HOURS, name="hour")
        stacked = sub.stack()
        stacked.name = f"on_{sanitize(u)}"
        target_frames.append(stacked)
    target_long = pd.concat(target_frames, axis=1).reset_index().rename(columns={"Run No": "run_no"})
    target_cols = [c for c in target_long.columns if c.startswith("on_")]

    id_cols = ["run_no", "split", "starttime", "timestamp", "hour"]
    feature_cols = [c for c in long_df.columns if c not in id_cols + ["day"]]

    train_df = long_df[long_df["split"] == "train"].merge(target_long, on=["run_no", "hour"], how="left")
    train_df = train_df[id_cols + feature_cols + target_cols]

    test_df = long_df[long_df["split"] == "test"][id_cols + feature_cols]

    assert train_df[target_cols].isna().sum().sum() == 0, "unexpected missing targets in train"
    assert len(train_df) == len(uc) * N_HOURS
    assert len(test_df) == len(pred_mapping) * N_HOURS

    train_df.to_parquet(f"{DATA}/processed/train_hourly.parquet", index=False)
    test_df.to_parquet(f"{DATA}/processed/test_hourly.parquet", index=False)

    print("train_df:", train_df.shape)
    print("test_df:", test_df.shape)
    print("\ncolumns:", train_df.columns.tolist())


if __name__ == "__main__":
    main()
