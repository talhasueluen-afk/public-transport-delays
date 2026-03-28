# Public Transport Delays — Predicting Delays with Weather & Events

A data science portfolio project analyzing and predicting public transport arrival delays
using weather conditions, city events, and traffic-related features.

> **Key finding:** Linear Regression failed to predict delays
> (R² = -0.0157), suggesting that weather and event data alone
> are insufficient — delays likely stem from operational factors
> not captured in this dataset.

---

## Objective

Investigate whether external factors (weather, events, traffic) can explain or predict
public transport arrival delays, using Exploratory Data Analysis and a Linear Regression model.

**Target variable:** `actual_arrival_delay_min` (continuous, in minutes)

---

## Methods

- **Data:** 2,000 trips, 24 features — Kaggle (khushikyad001/public-transport-delays-with-weather-and-events)
- **Preprocessing:** Filled 1,173 missing `event_type` values with `"No Event"`, one-hot encoded categoricals (`weather_condition`, `season`, `transport_type`, `event_type`), dropped ID columns and leaky features
- **Model:** `sklearn.linear_model.LinearRegression`, 80/20 train/test split (`random_state=42`)
- **Evaluation:** MAE, MSE, R²

---

## Key Results

**EDA**
- 74.95% of trips were delayed — delays are the norm in this dataset
- Weather condition had no significant impact — all categories showed similar delay rates (73–77%)
- Event type had no significant impact — all categories showed similar delay rates (72–77%)
- Arrival delays are broadly distributed between -3 and 29 minutes

**Model**

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | **7.73 minutes** |
| R² (R-squared) | **-0.0157** |

The model predicted approximately 13–15 minutes for nearly every trip regardless of input,
confirming that weather and event features carry almost no discriminative signal.
Linear Regression was not sufficient for this problem.

---

## Limitations

- Weather and event features show no meaningful variation in delay rates, making linear prediction ineffective
- R² below zero indicates the model does not outperform a simple mean baseline
- Likely causes of delays (vehicle condition, driver, route history) are not present in the dataset

---

## Next Improvements

- Engineer time-based features from `date` and `time` columns
- Try non-linear models (Random Forest, Gradient Boosting) to capture feature interactions
- Investigate additional data sources (operational logs, route history)
- Apply cross-validation for more robust evaluation

---

## Project Structure

```
public-transport-delays/
├── README.md
├── requirements.txt
├── public_transport_delays.ipynb   # EDA + Linear Regression model
├── model.pkl                       # Saved trained model
├── src/
│   ├── preprocess.py               # Data loading and preprocessing
│   └── train.py                    # Model training and evaluation
└── data/
    └── public_transport_delays.csv
```

---

## Technologies

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `Jupyter Notebook`

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open `public_transport_delays.ipynb` and run all cells top to bottom (`Kernel > Restart & Run All`).

> Ensure the dataset is at `data/public_transport_delays.csv` relative to the notebook.

---

*Project built as part of a data science portfolio.*
