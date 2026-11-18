import pandas as pd

SERVICE_FIX = {"exp.": "express", "sta.": "standard", "eco.": "economy"}


def clean_data(df):
    """Fix the data problems found in train.csv. Returns a new DataFrame."""
    df = df.copy()

    # 1. duplicate rows: keep the first copy
    df = df.drop_duplicates()

    # 2. unit mixup: Metro recorded weight in grams, convert to kg
    metro = df["destination"] == "Metro"
    df.loc[metro, "weight_kg"] = df.loc[metro, "weight_kg"] / 1000

    # 3. target typos: delivery_days typed 10 times too big
    typo = df["delivery_days"] > 15
    df.loc[typo, "delivery_days"] = df.loc[typo, "delivery_days"] / 10

    # 4. spelling variants in service: strip spaces, lowercase, expand short forms
    df["service"] = df["service"].str.strip().str.lower().replace(SERVICE_FIX)

    # 5. stuck sensor: declared_value_rs frozen at 19805 from 10 May to 27 May.
    #    Those readings are not real, so replace them with the median of the rest.
    stuck = (
        (df["declared_value_rs"] == 19805)
        & (df["date"] >= "2026-05-10")
        & (df["date"] <= "2026-05-27")
    )
    median_value = df.loc[~stuck, "declared_value_rs"].median()
    df.loc[stuck, "declared_value_rs"] = median_value

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/train.csv")
    out = clean_data(raw)
    print("rows before/after:", len(raw), len(out))
    print(out["service"].value_counts())
    print("weight_kg max:", out["weight_kg"].max())
    print("delivery_days max:", out["delivery_days"].max())
    print("declared 19805 left:", (out["declared_value_rs"] == 19805).sum())