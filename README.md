# Public Transport Delays — Predicting Delays with Weather & Events

A data science portfolio project that analyzes and attempts to predict public transport
delays using weather conditions, city events, and traffic-related features.

---

## Project Overview

This project explores whether external factors such as weather and city events can explain
or predict public transport arrival delays. The analysis includes exploratory data analysis
(EDA) and a Linear Regression model trained to predict `actual_arrival_delay_min`.

**Key conclusion:** The selected features did not provide sufficient predictive power for
a linear model — suggesting that delays in this dataset are not meaningfully explained
by weather or event data alone.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — khushikyad001/public-transport-delays-with-weather-and-events |
| Rows | 2,000 |
| Columns | 24 |
| Missing values | `event_type`: 1,173 NaN (filled with `"No Event"`) |

### Key Columns

| Column | Description |
|--------|-------------|
| `actual_arrival_delay_min` | Arrival delay in minutes — model target |
| `weather_condition` | Weather category at time of trip |
| `temperature_C` | Temperature in Celsius |
| `humidity_percent` | Humidity percentage |
| `wind_speed_kmh` | Wind speed in km/h |
| `precipitation_mm` | Precipitation in mm |
| `event_type` | Type of city event (Sports, Concert, etc.) |
| `event_attendance_est` | Estimated event attendance |
| `traffic_congestion_index` | Traffic congestion score (0–99) |
| `holiday` | Public holiday flag (0/1) |
| `peak_hour` | Peak hour flag (0/1) |
| `season` | Season of the year |
| `transport_type` | Bus, Metro, or Tram |
| `delayed` | Binary label — 1 if delayed (derived from target) |

---

## Project Structure

```
public-transport-delays/
├── claude.md                          # Project notes
├── README.md                          # This file
├── public transport delays.ipynb      # Main notebook: EDA + Linear Regression
└── data/
    └── public_transport_delays.csv    # Raw dataset
```

---

## EDA Findings

- **74.95% of trips were delayed** — delays are the norm rather than the exception in this dataset.
- **Weather condition had no significant impact** — all weather categories showed similar delay rates, ranging narrowly between 73% and 77%.
- **Event type had no significant impact** — delay rates across all event types (Sports, Concert, No Event, etc.) also fell within a similar range of 72–77%.
- **Arrival delays are broadly distributed** between -3 and 29 minutes, with no strong concentration around a single value.

These findings suggest that delays occur relatively uniformly across different weather and event conditions, making them difficult to predict from these features.

---

## Linear Regression Model

### Approach

1. Filled missing `event_type` values with `"No Event"`
2. One-hot encoded categorical features (`weather_condition`, `season`, `transport_type`, `event_type`)
3. Dropped ID columns, raw time strings, and leaky features (`actual_departure_delay_min`, `delayed`)
4. 80/20 train/test split (`random_state=42`)
5. Trained `sklearn.linear_model.LinearRegression`

### Results

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | **7.73 minutes** |
| R² (R-squared) | **-0.0157** |

### Interpretation

The R² of **-0.0157** means the model performs worse than simply predicting the mean delay
for every trip. In practice, the model predicted approximately 13–15 minutes for nearly all
trips regardless of the input features — reflecting the finding from EDA that weather and
event features carry almost no discriminative signal.

The MAE of **7.73 minutes** is close to the dataset's natural spread, confirming that the
model has not learned meaningful patterns beyond the overall average.

**Linear Regression was not sufficient for this problem** given these features.

### Possible Next Steps

- Engineer time-based features from the `date` and `time` columns
- Try non-linear models (Random Forest, Gradient Boosting) that can capture interactions
- Investigate features not present in the dataset (vehicle condition, driver, route history)
- Use cross-validation for more robust evaluation

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Model training and evaluation |

**Python version:** 3.x
**Environment:** Jupyter Notebook

---

## How to Run

1. **Clone or download** this repository.

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open** `public transport delays.ipynb` and run all cells top to bottom
   (`Kernel > Restart & Run All`).

> Make sure the dataset is located at `data/public_transport_delays.csv` relative to the notebook.

---

*Project built as part of a data science portfolio.*
