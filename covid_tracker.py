# COVID-19 Global Data Tracker

This notebook analyzes global COVID-19 trends, including cases, deaths, recoveries, and vaccinations across countries and time periods. The analysis uses the Our World in Data COVID-19 dataset.

## 1. Data Collection and Loading

First, we'll import necessary libraries and load the COVID-19 dataset.


```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
```


```python
# Load the dataset
# Note: In a live environment, you would use the downloaded CSV file
# df = pd.read_csv('owid-covid-data.csv')

# For demonstration, we'll use the direct URL to load the data
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
df = pd.read_csv(url)

print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
```

## 2. Data Exploration

Let's explore the dataset structure and understand what information we have.


```python
# Preview the first few rows
df.head()
```


```python
# Check column names
print("Column names in the dataset:")
for col in df.columns:
    print(f"- {col}")
```


```python
# Get basic information about the dataset
df.info()
```


```python
# Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Show columns with more than 0% missing values
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})
missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage (%)', ascending=False)
missing_df.head(10)  # Show top 10 columns with missing values
```


```python
# Check the range of dates in the dataset
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check unique countries/regions
print(f"Number of unique countries/regions: {df['location'].nunique()}")
print("\nSample of locations in the dataset:")
print(df['location'].unique()[:15])  # Display first 15 locations as a sample
```

## 3. Data Cleaning and Preparation

Now, let's clean and prepare the data for analysis.


```python
# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Create a copy of the cleaned dataset
clean_df = df.copy()

# Filter out aggregated regions like 'World', 'Europe', etc.
# We'll focus on individual countries for most of our analysis
continents = ['World', 'Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania', 'European Union']
countries_df = clean_df[~clean_df['location'].isin(continents)]

# Select a subset of interesting countries for detailed analysis
focus_countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'Russia', 'France', 'Germany', 'South Africa', 'Japan', 'Kenya']
focus_df = clean_df[clean_df['location'].isin(focus_countries)]

print(f"Filtered data to {len(focus_countries)} focus countries with {focus_df.shape[0]} total rows.")
```


```python
# Handle missing values for key metrics
# For simplicity, we'll forward-fill missing values within each country's timeline
key_metrics = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
               'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country]
    # Sort by date and forward fill values
    country_data = country_data.sort_values('date')
    for metric in key_metrics:
        if metric in country_data.columns:
            focus_df.loc[focus_df['location'] == country, metric] = country_data[metric].ffill()

# Calculate death rate (total_deaths / total_cases)
focus_df['death_rate'] = (focus_df['total_deaths'] / focus_df['total_cases']) * 100

# Calculate vaccination rate (people_fully_vaccinated / population)
focus_df['vaccination_rate'] = (focus_df['people_fully_vaccinated'] / focus_df['population']) * 100

print("Data cleaning completed. Added death_rate and vaccination_rate metrics.")
```

## 4. Exploratory Data Analysis (EDA)

### 4.1 COVID-19 Case Trends


```python
# Plot total cases over time for selected countries
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    plt.plot(country_data['date'], country_data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Plot total cases using log scale for better comparison
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    plt.plot(country_data['date'], country_data['total_cases'], label=country)

plt.yscale('log')
plt.title('Total COVID-19 Cases Over Time by Country (Log Scale)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Cases (Log Scale)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Compare daily new cases (7-day moving average to smooth daily fluctuations)
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    # Calculate 7-day moving average
    country_data['new_cases_smoothed'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['new_cases_smoothed'], label=country)

plt.title('Daily New COVID-19 Cases (7-day Moving Average)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('New Cases (7-day Avg)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4.2 COVID-19 Death Trends


```python
# Plot total deaths over time for selected countries
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)

plt.title('Total COVID-19 Deaths Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Deaths', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Plot total deaths using log scale
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)

plt.yscale('log')
plt.title('Total COVID-19 Deaths Over Time by Country (Log Scale)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Deaths (Log Scale)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Compare daily new deaths (7-day moving average)
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    # Calculate 7-day moving average
    country_data['new_deaths_smoothed'] = country_data['new_deaths'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['new_deaths_smoothed'], label=country)

plt.title('Daily New COVID-19 Deaths (7-day Moving Average)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('New Deaths (7-day Avg)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4.3 Death Rate Analysis


```python
# Calculate and plot death rates over time
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    plt.plot(country_data['date'], country_data['death_rate'], label=country)

plt.title('COVID-19 Death Rate Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Death Rate (Deaths/Cases %)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Bar chart of current death rates for focus countries
latest_data = focus_df.groupby('location').last().reset_index()
latest_data = latest_data.sort_values('death_rate', ascending=False)

plt.figure(figsize=(14, 8))
bars = plt.bar(latest_data['location'], latest_data['death_rate'], color=sns.color_palette("colorblind", len(focus_countries)))

plt.title('Current COVID-19 Death Rate by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Death Rate (Deaths/Cases %)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## 5. Vaccination Analysis


```python
# Plot cumulative vaccinations over time for selected countries
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    if 'total_vaccinations' in country_data.columns:
        plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)

plt.title('Total COVID-19 Vaccinations Over Time by Country', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Vaccinations', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Plot percentage of population fully vaccinated
plt.figure(figsize=(14, 8))

for country in focus_countries:
    country_data = focus_df[focus_df['location'] == country].sort_values('date')
    plt.plot(country_data['date'], country_data['vaccination_rate'], label=country)

plt.title('Percentage of Population Fully Vaccinated Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Population Fully Vaccinated (%)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)  # Set y-axis from 0-100%
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
# Bar chart of current vaccination rates for focus countries
latest_data = focus_df.groupby('location').last().reset_index()
latest_data = latest_data.sort_values('vaccination_rate', ascending=False)

plt.figure(figsize=(14, 8))
bars = plt.bar(latest_data['location'], latest_data['vaccination_rate'], color=sns.color_palette("viridis", len(focus_countries)))

plt.title('Current COVID-19 Vaccination Rate by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Population Fully Vaccinated (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 100)

# Add value labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```


```python
# Comparing vaccination progress with cases and deaths
# For this, we'll create a multi-faceted visualization showing relationships

# Define a function to create scatter plots for each country
def create_scatter_plot(country_name):
    country_data = focus_df[focus_df['location'] == country_name].copy()
    country_data = country_data.dropna(subset=['vaccination_rate', 'new_cases_smoothed', 'new_deaths_smoothed'])
    
    # Ensure we have all required columns with data
    if len(country_data) < 10:  # Skip if not enough data points
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(country_data['vaccination_rate'], 
                         country_data['new_cases_smoothed'], 
                         c=country_data['date'].astype(int) / 10**9,  # Color by date (convert to numeric)
                         cmap='viridis', 
                         alpha=0.7,
                         s=country_data['new_deaths_smoothed'] + 10)  # Size by deaths
    
    # Add colorbar to indicate dates
    cbar = plt.colorbar(scatter)
    cbar.set_label('Date Progression')
    
    plt.title(f'Vaccination vs. New Cases in {country_name}', fontsize=16)
    plt.xlabel('Vaccination Rate (%)', fontsize=12)
    plt.ylabel('New Cases (7-day avg)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    return fig

# Create scatter plots for a few selected countries (to avoid too many plots)
for country in ['United States', 'United Kingdom', 'Germany', 'Japan']:
    fig = create_scatter_plot(country)
    if fig:
        plt.tight_layout()
        plt.show()
```

## 6. Choropleth Map Visualization

Now, let's create a global choropleth map to visualize the current COVID-19 situation across countries.


```python
# Prepare data for the choropleth map
# We'll use the most recent data for each country
latest_global_data = df.groupby('location').last().reset_index()

# Keep only the necessary columns for our map
map_data = latest_global_data[['location', 'iso_code', 'total_cases', 'total_deaths', 
                               'total_cases_per_million', 'total_deaths_per_million',
                               'people_fully_vaccinated_per_hundred']].copy()

# Create choropleth map of total cases per million
fig = px.choropleth(map_data, 
                    locations="iso_code",
                    color="total_cases_per_million",
                    hover_name="location",
                    hover_data=["total_cases", "total_deaths", "total_deaths_per_million", "people_fully_vaccinated_per_hundred"],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="COVID-19 Total Cases per Million by Country")

fig.update_layout(
    title_font_size=20,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth'
    )
)

# Display the figure
fig.show()
```


```python
# Create choropleth map of vaccination rates
fig = px.choropleth(map_data, 
                    locations="iso_code",
                    color="people_fully_vaccinated_per_hundred",
                    hover_name="location",
                    hover_data=["total_cases_per_million", "total_deaths_per_million"],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="COVID-19 Vaccination Rates by Country (% Population Fully Vaccinated)")

fig.update_layout(
    title_font_size=20,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth'
    )
)

# Display the figure
fig.show()
```

## 7. Key Insights and Findings

Based on our analysis of the COVID-19 global data, here are the key insights:


```python
# Get latest summary statistics for our focus countries
latest_summary = latest_data.sort_values('total_cases', ascending=False)[
    ['location', 'total_cases', 'total_deaths', 'death_rate', 'vaccination_rate']
]

# Format the table for better readability
latest_summary['total_cases'] = latest_summary['total_cases'].map('{:,.0f}'.format)
latest_summary['total_deaths'] = latest_summary['total_deaths'].map('{:,.0f}'.format)
latest_summary['death_rate'] = latest_summary['death_rate'].map('{:.2f}%'.format)
latest_summary['vaccination_rate'] = latest_summary['vaccination_rate'].map('{:.2f}%'.format)

# Display the summary table
latest_summary.rename(columns={
    'location': 'Country',
    'total_cases': 'Total Cases',
    'total_deaths': 'Total Deaths',
    'death_rate': 'Death Rate',
    'vaccination_rate': 'Vaccination Rate'
}, inplace=True)

latest_summary
```

### Key Insights:

1. **Case Trends**: The United States, India, and Brazil have recorded the highest absolute numbers of COVID-19 cases globally among our focus countries. The pandemic showed multiple waves across different countries, with peaks occurring at different times.

2. **Death Rates**: There are significant variations in death rates (total deaths as a percentage of total cases) across countries. These differences may reflect variations in healthcare capacity, demographic factors, testing strategies, and reporting methodologies.

3. **Vaccination Rollout**: Countries demonstrated considerable differences in vaccination rates and rollout speeds. Some countries like the United Kingdom achieved high vaccination coverage relatively quickly, while others lagged behind.

4. **Vaccination Impact**: There appears to be a general inverse relationship between vaccination rates and new cases/deaths in many countries, particularly visible after vaccination campaigns gained momentum. This suggests the effectiveness of vaccines in reducing transmission and severe outcomes.

5. **Geographic Patterns**: The choropleth maps reveal distinct geographic patterns in COVID-19 impact and response, with certain regions experiencing disproportionate case loads and varying levels of vaccination coverage.

### Limitations of Analysis:

- Data reporting practices vary significantly between countries
- Testing capacity differences affect reported case numbers
- Population demographics and urbanization patterns influence disease spread
- Vaccine access has been unequal globally
- Some regions have limited data availability

## 8. Conclusion

This analysis provides a comprehensive overview of the global COVID-19 pandemic through data exploration and visualization. The pandemic has affected countries differently, with variations in infection rates, mortality, and vaccination progress.

Effective data analysis and visualization tools like those demonstrated in this notebook can help policymakers, healthcare professionals, and the general public understand pandemic trends and make informed decisions.

Future work could expand this analysis by:
- Incorporating more detailed demographic data
- Analyzing the impact of specific public health interventions
- Developing predictive models for future outbreaks
- Examining long-term economic and social impacts
- Comparing COVID-19 to historical pandemics

### References:
- Our World in Data COVID-19 Dataset
- World Health Organization COVID-19 Dashboard
- Johns Hopkins University Coronavirus Resource Center
