
import pandas as pd


def find_unit_mixup(df, column="weight_kg", group_col="destination"):
    """Find the group whose values are far bigger or smaller than others."""
    medians = df.groupby(group_col)[column].median()
    group_value = medians.idxmax()
    ratio = medians.max() / medians.drop(group_value).median()
    factor = min([10, 12, 60, 100, 1000], key=lambda f: abs(f - ratio))
    return group_value, factor


def find_target_typos(df, column="delivery_days", threshold=15):
    """Count rows whose target is about 10 times too big."""
    mask = df[column] > threshold
    return int(mask.sum())


def find_sentinel(df):
    """Check numeric columns for common sentinel values."""
    sentinel_values = [-999, -9999, 999999, 999999999]
    findings = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        for value in sentinel_values:
            count = int((df[column] == value).sum())

            if count > 0:
                findings[column] = {
                    "value": value,
                    "count": count
                }

    return findings


def find_spelling_variants(df, column="service"):
    """Map each spelling variant of a category to its real category.

    Returns a dict {variant: real_category}.
    """
    real_categories = ["express", "standard", "economy"]
    mapping = {}
    for raw in df[column].dropna().unique():
        clean = raw.strip().lower().rstrip(".")
        for category in real_categories:
            if len(clean) >= 3 and category.startswith(clean):
                mapping[raw] = category
                break
        else:
            mapping[raw] = "UNKNOWN"
    return mapping


def find_stuck_sensor(df, column="declared_value_rs", date_col="date"):
    """Find the longest run of rows (sorted by date) where a column repeats one value.

    Returns (column, stuck_value, start_date, end_date, run_length).
    """
    data = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    run_id = (data[column] != data[column].shift()).cumsum()
    run_lengths = run_id.value_counts()
    longest = run_lengths.idxmax()
    run = data[run_id == longest]
    start_date = str(run[date_col].min())[:10]
    end_date = str(run[date_col].max())[:10]
    return column, run[column].iloc[0], start_date, end_date, int(run_lengths.max())


def find_duplicates(df):
    """Return the total number of duplicate rows."""
    duplicate_count = int(df.duplicated().sum())
    return duplicate_count


def find_leaking_column(train, holdout):
    """
    Find columns present in training but missing from holdout
    that are highly correlated with the target.
    """
    target = "delivery_days"
    threshold = 0.99

    if target not in train.columns:
        raise ValueError("Target column is missing from training data.")

    missing_columns = [
        col for col in train.columns
        if col not in holdout.columns and col != target
    ]

    leaking_columns = {}

    for col in missing_columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            correlation = train[col].corr(train[target])

            if pd.notna(correlation) and abs(correlation) >= threshold:
                leaking_columns[col] = correlation

    return leaking_columns


if __name__ == "__main__":

    train = pd.read_csv("data/raw/train.csv")
    holdout = pd.read_csv("data/raw/holdout_features.csv")

    print(find_unit_mixup(train))
    print(find_target_typos(train))
    print("Sentinel Check:", find_sentinel(train))
    print("Service categories:", sorted(set(find_spelling_variants(train).values())))
    print("Stuck sensor:", find_stuck_sensor(train))
    print("Duplicate rows:", find_duplicates(train))
    print("Leaking columns:", find_leaking_column(train, holdout))
