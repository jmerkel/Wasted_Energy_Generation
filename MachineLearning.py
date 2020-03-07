# Import Libraries
import pandas as pd

# Library Settings
pd.set_option('display.max_columns', None)

# Import Data
weather_path="Resources/weather_features.csv"
weather_df=pd.read_csv(weather_path)
weather_df.head()

energy_path="Resources/energy_dataset.csv"
energy_df=pd.read_csv(energy_path)
energy_df.head()

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

print(len(Valencia_df),len(Madrid_df),len(Bilbao_df),len(Barcelona_df),len(Seville_df))


## Energy Frames
energy_df = energy_df[pd.notnull(energy_df["generation waste"])]

# Determine cutoff point for Excessive Wasted Energy
print(energy_df["generation waste"].describe())

# Prep Dataframe - 310 MW is cutoff (Describe 75% showed 310 MW)
energy_df["excessive waste"] = energy_df["generation waste"].\
    map(lambda x: 1 if x > 310.0 else 0)
print(energy_df[["generation waste", "excessive waste"]].tail(10))
print(energy_df["excessive waste"].value_counts())

# Initial Code testing to be done with Madrid DF
# Combine with Weather
# Machine Learning
# Output