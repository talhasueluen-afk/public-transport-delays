import pandas as pd


def load_and_preprocess(filepath="data/public_transport_delays.csv"):
    """
    Load the dataset, handle missing values, encode categoricals,
    and return features (X) and target (y) ready for model training.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series):    Target variable (actual_arrival_delay_min).
    """

    # --- Load ---
    df = pd.read_csv(filepath)

    # --- Handle missing values ---
    # event_type has 1,173 missing values — trips with no event get labelled "No Event"
    df["event_type"] = df["event_type"].fillna("No Event")

    # --- One-hot encode categorical columns ---
    # drop_first=True avoids the dummy variable trap (one redundant column per category)
    df = pd.get_dummies(
        df,
        columns=["weather_condition", "season", "transport_type", "event_type"],
        drop_first=True,
    )

    # --- Drop columns that are not useful for prediction ---
    drop_cols = [
        "trip_id",                    # unique identifier, no predictive value
        "date", "time",               # raw strings, not parsed into features
        "route_id",                   # high-cardinality ID
        "origin_station",             # high-cardinality string
        "destination_station",        # high-cardinality string
        "scheduled_departure",        # raw time string
        "scheduled_arrival",          # raw time string
        "actual_departure_delay_min", # directly correlated with target — data leakage
        "actual_arrival_delay_min",   # this IS the target
        "delayed",                    # derived from target — data leakage
    ]
    df = df.drop(columns=drop_cols)

    # --- Split into features and target ---
    y = pd.read_csv(filepath)["actual_arrival_delay_min"]
    X = df

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nFeature columns:\n{X.columns.tolist()}")
