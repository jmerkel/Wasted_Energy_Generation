# Import Libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import sqlite3
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import session
from sqlalchemy import create_engine, func
import plotly.express as px
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def import_data(path):
    return pd.read_csv(path)

def create_city(weather_df, CityName):
    df= weather_df.loc[weather_df["city_name"] == CityName]
    # Prep Weather Table for Merge
    # Drop non needed columns (descriptions, weather Description (ID code remains), )
    city_df = df.drop_duplicates(keep=False, inplace=False)
    city_df = city_df.drop(["city_name", "weather_main", "weather_description", "weather_icon"], 1)
    return city_df

def clean_energy_table(energy_df):
    # Determine cutoff point for Excessive Wasted Energy
    energy_df = energy_df[pd.notnull(energy_df["generation waste"])]
    print ("Describe the wasted energy valeues")
    print(energy_df["generation waste"].describe())
    print ("310 mW used as it describes upper quartile") 

    # Prep Dataframe - 310 MW is cutoff (Describe 75% showed 310 MW)
    energy_df["excessive waste"] = energy_df["generation waste"].\
    map(lambda x: 1 if x > 310.0 else 0)

    # Remove NaN fields
    print("\nFind NaN energy items")
    for column in energy_df.columns:
        print ([column, energy_df[column].isnull().sum()])

    # Data Empty from source - All values 0 - Determined from source values
    energy_clean_df = energy_df.drop(['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead'], 1)
    energy_clean_df = energy_clean_df.dropna()
    energy_clean_df = energy_clean_df.rename(columns={"time": "dt_iso"})

    return energy_clean_df

def check_column(df, df_name):
    print(df_name + " Column Null Values")
    for column in df.columns:
        print ([column, df[column].isnull().sum()])
    print("\n")

def merge_tables(city_df, energy_df):
    City_Weather_Data_df = city_df.merge(energy_forecast_df, on="dt_iso") 
    City_Weather_Data_df["dt_iso"] = pd.to_datetime(City_Weather_Data_df["dt_iso"], utc=True, infer_datetime_format=True)
    City_Weather_Data_df["dt_iso"] = pd.to_datetime(City_Weather_Data_df["dt_iso"]).astype(int)/10**9
    return City_Weather_Data_df

def write_SQL (engine, conn, df, df_name):
    df.to_sql(df_name, conn, if_exists='replace', index=False)
      
# Import Data
weather_df = import_data("Resources/weather_features.csv")
energy_df = import_data("Resources/energy_dataset.csv")

# Preprocessing
print("Find all Unique city Names") 
print(weather_df["city_name"].unique())
print("\n")

## Barcelona has a space in front of the string
weather_df["city_name"] = weather_df["city_name"].str.strip() #Remove leading & trailing spaces

# Weather Frames - Isolate the dataframes by city_name
Madrid_df = create_city(weather_df, "Madrid")
Barcelona_df = create_city(weather_df, "Barcelona")
Valencia_df = create_city(weather_df, "Valencia")
Seville_df = create_city(weather_df, "Seville")
Bilbao_df = create_city(weather_df, "Bilbao")

print("Find all Unique city Names-Fix") 
print(weather_df["city_name"].unique())
print("\n")

print(Barcelona_df.head())
# Energy Frames
energy_clean_df = clean_energy_table(energy_df)

# Ensure all tables have clean columns
check_column(energy_clean_df, "Energy Table")
check_column(weather_df, "Weather Table")

## Prep Tables for Processing
# Prep Energy Table for Merge
energy_forecast_df = energy_clean_df[[
    "dt_iso", 
    "forecast solar day ahead", 
    "forecast wind onshore day ahead", 
    "total load forecast",
    "price day ahead",
    "excessive waste"]].copy()

# Review Data Types 
print("\nMadrid Data Types")
print(Madrid_df.dtypes)
print("\nEnergy Forecast Types")
print(energy_forecast_df.dtypes)

# Merge city weather tables with energy tables
Madrid_Weather_Data_df = merge_tables(Madrid_df, energy_forecast_df)
Barcelona_Weather_Data_df = merge_tables(Barcelona_df, energy_forecast_df)
Valencia_Weather_Data_df = merge_tables(Valencia_df, energy_forecast_df)
Seville_Weather_Data_df = merge_tables(Seville_df, energy_forecast_df)
Bilbao_Weather_Data_df = merge_tables(Bilbao_df, energy_forecast_df)

# Verify All columns are not objects
print("\nColumn Types - Verification")
print(Madrid_Weather_Data_df.dtypes)


#SQLAlchemy & SQLite
## Write Data - SQLite
engine = create_engine('sqlite:///Resources/energy_data.sqlite', echo=False)
conn = sqlite3.connect('Resources/energy_data.sqlite')

write_SQL(engine, conn, Madrid_Weather_Data_df, "forecast_Madrid")
write_SQL(engine, conn, Barcelona_Weather_Data_df, "forecast_Barcelona")
write_SQL(engine, conn, Valencia_Weather_Data_df, "forecast_Valencia")
write_SQL(engine, conn, Seville_Weather_Data_df, "forecast_Seville")
write_SQL(engine, conn, Bilbao_Weather_Data_df, "forecast_Bilbao")


#### READ SQLite DB ####
#Base = automap_base()
#engine = create_engine("sqlite:///energy.sqlite")
#Base.prepare(engine, reflect=True)
#Base.classes.keys() # View Classes found by automap
#session = Session(engine) # Allow query

Madrid_df = pd.read_sql('select * from forecast_Madrid', conn)
Barcelona_df = pd.read_sql('select * from forecast_Barcelona', conn)
Valencia_df = pd.read_sql('select * from forecast_Valencia', conn)
Seville_df = pd.read_sql('select * from forecast_Seville', conn)
Bilbao_df = pd.read_sql('select * from forecast_Bilbao', conn)

print("Read DataFrame")
print(Madrid_df.head())

# Machine Learning Portion - Supervised Learning

#def remove() :
    #import datetime as dt
    #from datetime import timezone
    
    ### Madrid with only weather ###
    #Madrid_Weather_Data_v2_df = Madrid_Weather_Data_df.drop([
    #    "forecast solar day ahead",
    #    "forecast wind onshore day ahead",
    #    "total load forecast",
    #    "price day ahead",
    #    "weather_id"],1)
    
    
    # Unsupervised Learning
    ### Verify Excessive waste column is 0 or 1
    #print("Excessive Waste Unique Values")
    #print(Madrid_Weather_Data_df["excessive waste"].unique())
    #
    #model=KMeans(n_clusters=3,random_state=5)
    #model.fit(Madrid_Weather_Data_df)
    ## Check the relation between pressure, temperature by the class of excessive waste
    #Madrid_Weather_Data_df.hvplot.scatter("temp",y="pressure",by="excessive waste")
    #
    ##The plot above tell people that the majority of the excessive waste happened when the pressure goes high. 
    ##From that, we can probably had a clue this distribuction can lead to some clue. 
    ##Let's try to run the K-means test with different clusters.
    #
    #def test_cluster_amount(df,clusters):
    #    model=KMeans(n_clusters=clusters,random_state=5)
    #    model
    #    model.fit(df)
    #    df["excessive waste"]=model.labels_
    #
    #def get_clusters(k,data):
    #    model=KMeans(n_clusters=k,random_state=0)
    #    model.fit(data)
    #    predictions=model.predict(data)
    #    data["excessive waste"]=model.labels_
    #    return data
    #
    ## only 1 cluster
    #Madrid_Weather_Data_df.hvplot.scatter(x="temp",y="pressure")
    #
    ##K-Means prediction of 2 clusters
    #test_cluster_amount(Madrid_Weather_Data_df,2)
    #Madrid_Weather_Data_df.hvplot.scatter("temp",y="pressure",by="excessive waste")
    #
    ##test_cluster_amount(Madrid_Weather_Data_df,3)
    ##Madrid_Weather_Data_df.hvplot.scatter("temp",y="pressure",by="excessive waste")
    #
    ### This plot does not tell anythings, still needs to optimize
    #fig=px.scatter_3d(Madrid_Weather_Data_df,
    #                 x="temp",
    #                 y="pressure",
    #                 z="humidity",
    #                  color="excessive waste",
    #                  symbol="excessive waste",
    #                  size="temp",
    #                  width=800,
    #                  )
    #fig.update_layout(legend=dict(x=0,y=1))
    #fig.show()
    #
    #inertia=[]
    #k=list(range(1,11))
    #for i in k:
    #    km=KMeans(n_clusters=i,random_state=0)
    #    km.fit(Madrid_Weather_Data_df)
    #inertia.append(km.inertia_)
    #
    #two_clusters=get_clusters(2,Madrid_Weather_Data_df)
    #two_clusters.head()
    #
    #Madrid_Weather_Data_df.hvplot.scatter("temp",y="pressure",by="excessive waste")