import pandas as pd


def find_unit_mixup(df, column="weight_kg", group_col="destination"):
    """Find the group whose values are far bigger or smaller than the others.

    Returns (group_value, factor).
    """
    medians = df.groupby(group_col)[column].median()
    group_value = medians.idxmax()
    ratio = medians.max() / medians.drop(group_value).median()
    factor = min([10, 12, 60, 100, 1000], key=lambda f: abs(f - ratio))
    return group_value, factor


def find_target_typos(df, column="delivery_days", threshold=15):
    """Return the number of rows whose target is about 10 times too big."""
    mask = df[column] > threshold
    return int(mask.sum())

def find_sentinel(df):
    """
    Check numeric columns for common sentinel values.
    Returns a dictionary of detected sentinel values.
    """

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



if __name__ == "__main__":
    df = pd.read_csv("data/raw/train.csv")

    print(find_unit_mixup(df))
    print(find_target_typos(df))
    print("Sentinel Check:", find_sentinel(df))
    