import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import load_and_preprocess


def train(filepath="data/public_transport_delays.csv", model_output="model.pkl"):
    """
    Load preprocessed data, train a Linear Regression model,
    print evaluation metrics, and save the model to disk.

    Parameters:
        filepath (str):     Path to the raw CSV dataset.
        model_output (str): Path where the trained model will be saved.
    """

    # --- Load and preprocess data ---
    X, y = load_and_preprocess(filepath)

    # --- Train/test split ---
    # 80% of data for training, 20% held out for evaluation
    # random_state=42 ensures the split is reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set:  {X_train.shape[0]} rows")
    print(f"Test set:      {X_test.shape[0]} rows")

    # --- Train the model ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Generate predictions on the test set ---
    y_pred = model.predict(X_test)

    # --- Evaluate ---
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"MAE  (Mean Absolute Error):  {mae:.2f} minutes")
    print(f"MSE  (Mean Squared Error):   {mse:.2f}")
    print(f"R²   (R-squared):            {r2:.4f}")

    # --- Save the trained model to disk ---
    with open(model_output, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_output}")


if __name__ == "__main__":
    train()
