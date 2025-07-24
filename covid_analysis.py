# Done by Nader Mohamed Elsaid Salama                 ONL3_AIS3_G1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataprocessing:
    def __init__(self, url):
        """
        Initialize with the data URL and load the data.
        """
        self.url = url
        self.df = None

    def load_data(self):
        """
        Loads the dataset from the URL, cleans it, and filters for countries only.
        """
        # Load data
        self.df = pd.read_csv(self.url)

        # Peek at data
        print(self.df.head())
        print(self.df.info())

        # Convert date column to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Fill missing new_cases/new_deaths with 0
        self.df['new_cases'] = self.df['new_cases'].fillna(0)
        self.df['new_deaths'] = self.df['new_deaths'].fillna(0)

        # Filter out rows where location is not a country (exclude World, continents)
        exclude_locations = ["World", "Africa", "Asia", "Europe", "European Union",
                             "International", "North America", "Oceania", "South America"]
        self.df = self.df[~self.df['location'].isin(exclude_locations)]

        print("Data loaded and cleaned.")

    def get_country_data(self, country):
        """
        Returns a DataFrame for a specific country.
        """
        country_df = self.df[self.df['location'] == country]
        return country_df


if __name__ == "__main__":
    # URL provided in the assignment
    DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

    processor = Dataprocessing(DATA_URL)
    processor.load_data()

    # Test: Get data for Egypt
    country_name = "Egypt"
    egypt_data = processor.get_country_data(country_name)

    # Calculate daily increase in cases
    egypt_data['cumulative_cases'] = egypt_data['new_cases'].cumsum()

    # Calculate % of population vaccinated
    egypt_data['perc_vaccinated'] = (egypt_data['people_vaccinated'] / egypt_data['population']) * 100

    # Plot 1: Cases over Time using Line Chart
    plt.figure(figsize=(10, 6))
    plt.plot(egypt_data['date'], egypt_data['cumulative_cases'], label="Cumulative Cases")
    plt.title(f"COVID-19 Cumulative Cases Over Time in {country_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cases_over_time.png")
    plt.close()

    # Plot 2: Comparison Bar Chart of Total Deaths 
    countries = ["Egypt", "Italy", "India", "Brazil", "United States"]
    total_deaths = []
    for c in countries:
        df_c = processor.get_country_data(c)
        total_deaths.append(df_c['total_deaths'].max())

    plt.figure(figsize=(8, 6))
    plt.bar(countries, total_deaths, color='red')
    plt.title("Total Deaths Comparison")
    plt.xlabel("Country")
    plt.ylabel("Total Deaths")
    plt.tight_layout()
    plt.savefig("total_deaths_comparison.png")
    plt.close()

    # Plot 3: Vaccination Progress in Egypt 
    plt.figure(figsize=(10, 6))
    plt.plot(egypt_data['date'], egypt_data['perc_vaccinated'], color='green', label="% Vaccinated")
    plt.title(f"COVID-19 Vaccination Progress in {country_name}")
    plt.xlabel("Date")
    plt.ylabel("% of Population Vaccinated")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vaccination_progress.png")
    plt.close()

    print("All charts saved: cases_over_time.png, total_deaths_comparison.png, vaccination_progress.png")