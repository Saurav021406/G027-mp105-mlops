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


if __name__ == "__main__":
    df = pd.read_csv("data/raw/train.csv")
    print(find_unit_mixup(df))