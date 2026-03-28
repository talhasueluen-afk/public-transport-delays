# Public Transport Delays — Predicting Delays with Weather & Events

## Project Overview
A data science project that analyzes and predicts public transport delays
based on external factors such as weather conditions and city events.

Dataset: public_transport_delays.csv (2000 rows, 24 columns)
Source: Kaggle — khushikyad001/public-transport-delays-with-weather-and-events

## Goals
- Explore how weather conditions (rain, snow, temperature) affect delays
- Analyze whether major city events increase delay probability
- Build a Linear Regression model to predict delay duration

## Tech Stack
- Python 3.x
- Jupyter Notebook
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)

## Project Structure
public-transport-delays/
├── CLAUDE.md
├── data/
│   └── public_transport_delays.csv
├── notebooks/
│   └── 01_EDA.ipynb
└── README.md

## Dataset Columns
- trip_id, date, time, transport_type, route_id
- origin_station, destination_station
- scheduled_departure, scheduled_arrival
- actual_departure_delay_min, actual_arrival_delay_min
- weather_condition, temperature_C, humidity_percent
- wind_speed_kmh, precipitation_mm
- event_type, event_attendance_est
- traffic_congestion_index
- holiday, peak_hour, weekday, season
- delayed (target variable)

## Developer Notes
- Beginner-friendly: explain each step clearly
- Language: English (code + comments + outputs)
- Proceed step by step, one notebook cell at a time