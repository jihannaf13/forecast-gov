import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the forecast results
with open('models/forecasts.json', 'r') as f:
    forecasts = json.load(f)

with open('models/model_performance.json', 'r') as f:
    performance = json.load(f)

# Load processed data for context
national_data = pd.read_csv('processed_data/national_timeseries.csv')
provincial_data = pd.read_csv('processed_data/complete_cleaned_data.csv')

print("DETAILED ANALYSIS")
print("="*50)

# 1. Identify causes of decline
print("\nPRODUCTION DECLINE ANALYSIS:")     
decline_factors = national_data.copy()
decline_factors['Area_change'] = decline_factors['Luas_Panen_ha'].pct_change() * 100
decline_factors['Productivity_change'] = decline_factors['Produktivitas_ku_ha'].pct_change() * 100
decline_factors['Production_change'] = decline_factors['Produksi_ton'].pct_change() * 100

print(decline_factors[['Year', 'Area_change', 'Productivity_change', 'Production_change']].round(2))

# 2. Provincial contribution analysis
print("\nPROVINCIAL IMPACT ANALYSIS:")
provincial_trends = provincial_data.groupby(['Province', 'Year'])['Produksi_ton'].sum().unstack(fill_value=0)

# Filter out provinces that had zero production in 2018 to avoid division by zero
provinces_with_2018_data = provincial_trends[provincial_trends[2018] > 0]
provincial_change = ((provinces_with_2018_data[2024] - provinces_with_2018_data[2018]) / provinces_with_2018_data[2018] * 100).sort_values()

print("Top 5 Declining Provinces:")
print(provincial_change.head().round(1))
print("\nTop 5 Growing Provinces:")
print(provincial_change.tail().round(1))

# Show new provinces separately
new_provinces = provincial_trends[provincial_trends[2018] == 0]
if len(new_provinces) > 0:
    print(f"\nNew Provinces (established after 2018):")
    for province in new_provinces.index:
        production_2024 = new_provinces.loc[province, 2024]
        if production_2024 > 0:
            print(f"{province}: {production_2024:,.0f} tons in 2024")

# 3. Forecast reliability assessment
ensemble_forecast = forecasts['ensemble_Produksi_ton']
print(f"\nFORECAST RELIABILITY:")
print(f"Model R²: {ensemble_forecast['metrics']['R2']:.3f}")
print(f"RMSE: {ensemble_forecast['metrics']['RMSE']:,.0f} tons")
print(f"Uncertainty: ±{(ensemble_forecast['metrics']['RMSE']/106285453)*100:.1f}%")

print("\nRECOMMENDATIONS:")
print("1. Investigate area reduction causes (policy, climate, economics)")
print("2. Focus on productivity improvement programs")
print("3. Monitor top declining provinces for intervention")
print("4. Validate forecasts with external factors (weather, policy changes)")