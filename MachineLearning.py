# Import Libraries
import pandas as pd
pd.set_option('display.max_columns', None)

# Import Data
weather_path="Resources/weather_features.csv"
weather_df=pd.read_csv(weather_path)

energy_path="Resources/energy_dataset.csv"
energy_df=pd.read_csv(energy_path)

### Preprocessing
weather_df["city_name"].unique()

## Weather Frames
# Isolate the dataframes by city_name
Valencia_df=weather_df.loc[weather_df["city_name"]=="Valencia"]
Madrid_df=weather_df.loc[weather_df["city_name"]=="Madrid"]
Barcelona_df=weather_df.loc[weather_df["city_name"]==' Barcelona']
Seville_df=weather_df.loc[weather_df["city_name"]=='Seville']
Bilbao_df=weather_df.loc[weather_df["city_name"]=="Bilbao"]

Valencia_df.drop_duplicates(keep=False,inplace=True)
Madrid_df.drop_duplicates(keep=False,inplace=True)
Bilbao_df.drop_duplicates(keep=False,inplace=True)
Barcelona_df.drop_duplicates(keep=False,inplace=True)
Seville_df.drop_duplicates(keep=False,inplace=True)


## Energy Frames
# Determine cutoff point for Excessive Wasted Energy
energy_df = energy_df[pd.notnull(energy_df["generation waste"])]
print(energy_df["generation waste"].describe()) 

# Prep Dataframe - 310 MW is cutoff (Describe 75% showed 310 MW)
energy_df["excessive waste"] = energy_df["generation waste"].\
    map(lambda x: 1 if x > 310.0 else 0)

print("\nFind NaN energy items")
for column in energy_df.columns:
    print ([column, energy_df[column].isnull().sum()])

energy_clean_df = energy_df.drop(['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead'], 1)
energy_clean_df = energy_clean_df.dropna()

print("\n")
print("Energy Table Columns")
for column in energy_clean_df.columns:
    print ([column, energy_clean_df[column].isnull().sum()])


# Initial Code testing to be done with Madrid DF
print("\n")
print("Madrid Weather Tabl Columns")
for column in Madrid_df.columns:
    print ([column, Madrid_df[column].isnull().sum()])

## Prep Tables for Processing
# Prep Weather Table for Merge
# Drop non needed columns (descriptions, weather Description (ID code remains), )
Madrid_Prep_df = Madrid_df.drop(["city_name", "weather_main", "weather_description", "weather_icon"], 1)

# Prep Energy Table for Merge
energy_clean_df = energy_clean_df.rename(columns={"time": "dt_iso"})
energy_forecast_df = energy_clean_df[[
    "dt_iso", 
    "forecast solar day ahead", 
    "forecast wind onshore day ahead", 
    "total load forecast",
    "price day ahead",
    "excessive waste"]].copy()

# Merge Madrid with weather & energy forecasts
print(Madrid_Prep_df.dtypes)
print()
print(energy_forecast_df.dtypes)

# Inner Join
Madrid_Weather_Data_df = Madrid_Prep_df.merge(energy_forecast_df, on="dt_iso") 
Madrid_Weather_Data_df["dt_iso"] = pd.to_datetime(Madrid_Weather_Data_df["dt_iso"], utc=True, infer_datetime_format=True)

print("\nColumn Types")
print(Madrid_Weather_Data_df.dtypes)
print("\nCheck Values")
print(Madrid_Weather_Data_df.head())


# Madrid with only weather
Madrid_Weather_Data_v2_df = Madrid_Weather_Data_df.drop([
    "forecast solar day ahead",
    "forecast wind onshore day ahead",
    "total load forecast",
    "price day ahead",
    "weather_id"],1)

# Machine Learning