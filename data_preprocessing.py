import pandas as pd
import numpy as np
import requests


def fetch_data():
    """
    Fetch precipitation, soil moisture, and evapotranspiration data.
    Currently loads a placeholder CSV file.
    Replace this with real API calls in the future.
    """

    weather_data = pd.read_csv("weather_data.csv", skiprows=10)
    return weather_data


def preprocess_data(weather_data):
    """
    Clean and prepare the data before sending it to the ML model.
    """

    # Fill missing values using forward fill
    weather_data = weather_data.ffill()

    processed_data = weather_data

    return processed_data


def save_cleaned_data(processed_data):
    """
    Save cleaned data for later use.
    """

    processed_data.to_csv(
        "cleaned_data/weather_data_cleaned.csv",
        index=False
    )


if __name__ == "__main__":

    raw_data = fetch_data()
    processed_data = preprocess_data(raw_data)

    save_cleaned_data(processed_data)
