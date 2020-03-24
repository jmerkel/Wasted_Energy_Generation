# Import Libraries
import pandas as pd
pd.set_option('display.max_columns', None)

import sqlite3
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from sqlalchemy.types import Integer, Text, String, DateTime, Float
from sqlalchemy import join
from sqlalchemy.sql import select
import plotly.express as px
import hvplot.pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC


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
    print ("\n310 mW used as it describes upper quartile") 

    # Prep Dataframe - 310 MW is cutoff (Describe 75% showed 310 MW)
    energy_df["excessive waste"] = energy_df["generation waste"].\
    map(lambda x: 1 if x > 310.0 else 0)

    # Remove NaN fields
    print("Find NaN energy items")
    for column in energy_df.columns:
        print ([column, energy_df[column].isnull().sum()])

    # Data Empty from source - All values 0 - Determined from source values
    energy_clean_df = energy_df.drop(['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead'], 1)
    energy_clean_df = energy_clean_df.dropna()
    energy_clean_df = energy_clean_df.rename(columns={"time": "dt_iso"})

    return energy_clean_df

def check_column(df, df_name):
    print("\n" + df_name + " Column Null Values")
    for column in df.columns:
        print ([column, df[column].isnull().sum()])
    print("\n")

def merge_tables(city_df, energy_df):
    City_Weather_Data_df = city_df.merge(energy_forecast_df, on="dt_iso") 
    City_Weather_Data_df["dt_iso"] = pd.to_datetime(City_Weather_Data_df["dt_iso"], utc=True, infer_datetime_format=True)
    City_Weather_Data_df["dt_iso"] = pd.to_datetime(City_Weather_Data_df["dt_iso"]).astype(int)/10**9
    return City_Weather_Data_df

def write_SQL (engine, conn, df, df_name):
    new_columns = [column.replace(' ', '_').lower() for column in df]
    df.columns = new_columns
    df.to_sql(df_name,
                con=engine, 
                if_exists='replace',
                #index = False, 
                dtype={"dt_iso": Float(),
                    "temp": Float(),
                    "temp_min": Float(),
                    "temp_max": Float(),
                    "pressure": Integer(),
                    "humidity": Integer(), 
                    "wind_speed": Integer(),
                    "wind_deg": Integer(),
                    "rain_1h": Float(),
                    "rain_3h": Float(),
                    "snow_3h": Float(),
                    "clouds_all": Integer(),
                    "weather_id": Integer(),
                    "forecast_solar_day_ahead": Float(),
                    "forecast_wind_onshore_day_ahead": Float(),
                    "total_load_forecast": Float(),
                    "price_day_ahead": Float(),
                    "excessive_waste": Integer()})

    #df.to_sql(df_name, conn, if_exists='replace', index=False)

def sql_join(engine):
    ### SQLite Join Function
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    print(Base.classes.keys()) # No Data? -- No primary Key
    Forecast_Madrid = Base.classes.forecast_Madrid
    Forecast_Barcelona = Base.classes.forecast_Barcelona
    session = Sesion(engine)

    #result = session.query(Forecast_Madrid).join(Forecast_Barcelona).all() # Join all items

def model_output(results, y_test, y_pred):
    print(results.head())
    print("\nAccuracy Score: " + str(accuracy_score(y_test, y_pred)))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    #Classification Report - Provides precision, recall, F1 score
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

def model_test(df, df2):
    # Machine Learning Portion - Supervised Learning
    ### Logistic Regression
    ## 17.3.1 - 17.3.3 for code (17.3.2)
    ## Madrid = Train
    ## 17.4.1 - Accuracy, Precision, Sensitivity
    ## 17.4.2 - Confusion Matrix

    ### Support Vector Machine (SVM)
    ## 17.5.2 - SVM

    ## 17.6.4 - Scale & Normalize Data
    ## 17.7.2 - Decision Tree

    y = df["excessive_waste"]
    X = df.drop(columns="excessive_waste")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

    #Scale Data
    scaler = StandardScaler()
    # Fitting the Standard Scaler with the training data.
    X_scaler = scaler.fit(X_train)
    # Scaling the data.
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    #Logistic Regression
    print ("\n\nLogistic Regression")
    print ("X_Train Shape")
    print(X_train.shape)
    classifier = LogisticRegression(solver='lbfgs', max_iter=200,random_state=1)
    classifier.fit (X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    results = pd.DataFrame({
        "Prediction": y_pred, 
        "Actual": y_test}).reset_index(drop=True)
    model_output(results, y_test, y_pred)

    ### SVM
    # Instantiate a linear SVM model
    print("\n\nLinear SVM")
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({    
        "Prediction": y_pred, 
        "Actual": y_test
    }).reset_index(drop=True)
    model_output(results, y_test, y_pred)
    

    # Instantiate a poly SVM model
    print("\n\nSVM Poly")
    model = SVC(kernel='poly') 
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({    
        "Prediction": y_pred, 
        "Actual": y_test
    }).reset_index(drop=True)
    model_output(results, y_test, y_pred)

    # Instantiate a rbf SVM model
    print("\n\nRBF SVM")
    model = SVC(kernel='rbf') #Is the orientation of the hyperplane linear or non linear?
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({    
        "Prediction": y_pred, 
        "Actual": y_test
    }).reset_index(drop=True)
    model_output(results, y_test, y_pred)

    ### Production Test
    y = df["excessive_waste"]
    X = df.drop(columns="excessive_waste")
    y_test = df2["excessive_waste"]
    X_test = df2.drop(columns="excessive_waste")

    scaler = StandardScaler()
    # Fitting the Standard Scaler with the training data.
    X_scaler = scaler.fit(X)
    # Scaling the data.
    X_scaled = X_scaler.transform(X)
    X_test_scaled = X_scaler.transform(X_test)

    print("\n\nRBF SVM - Complete Test - Madrid vs Barcelona")
    model = SVC(kernel='rbf') #Is the orientation of the hyperplane linear or non linear?
    model.fit(X_scaled, y)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({    
        "Prediction": y_pred, 
        "Actual": y_test
    }).reset_index(drop=True)
    model_output(results, y_test, y_pred)

def city_compare(df, df2, df2_name):
    #Training Set
    y = df["excessive_waste"] 
    X = df.drop(columns="excessive_waste")

    #Test Set
    y_test = df2["excessive_waste"]
    X_test = df2.drop(columns="excessive_waste")

    # Fitting the Standard Scaler with the training data.
    scaler = StandardScaler()
    X_scaler = scaler.fit(X)
    # Fitting the Standard Scaler with the training data.
    X_scaled = X_scaler.transform(X)
    X_test_scaled = X_scaler.transform(X_test)

    section_name = "\n\nSVM RBF - " + df2_name

    print(section_name)
    model = SVC(kernel='rbf') # Non Linear Hyperplane
    model.fit(X_scaled, y)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({    
        "Prediction": y_pred, 
        "Actual": y_test
    }).reset_index(drop=True)

    model_output(results, y_test, y_pred) 
    report = classification_report(y_test, y_pred, output_dict=True)
    out_df = pd.DataFrame(report).transpose()
    out_df.to_csv("Output/" + df2_name + ".csv", index=True)

def generate_Pics(df):
    ### Generate Pic
    # Scatter Plot
    plot = Madrid_df.hvplot.scatter("temp",y="pressure",by="excessive waste")
    hvplot.show(plot)

    plot = Madrid_df.hvplot.scatter("humidity",y="pressure",by="excessive waste")
    hvplot.show(plot)

    plot = Madrid_df.hvplot.scatter("wind_speed",y="wind_deg",by="excessive waste")
    hvplot.show(plot)

# def temp():
### MAIN ####
# Import Data
weather_df = import_data("Resources/weather_features.csv")
energy_df = import_data("Resources/energy_dataset.csv")

# Preprocessing
print("Find all Unique city Names") 
print(weather_df["city_name"].unique())

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
print("Madrid Data Types")
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
engine = create_engine("sqlite:///Resources/energy_data.sqlite", echo=False)
conn = sqlite3.connect('Resources/energy_data.sqlite')

write_SQL(engine, conn, Madrid_Weather_Data_df, "forecast_Madrid")
write_SQL(engine, conn, Barcelona_Weather_Data_df, "forecast_Barcelona")
write_SQL(engine, conn, Valencia_Weather_Data_df, "forecast_Valencia")
write_SQL(engine, conn, Seville_Weather_Data_df, "forecast_Seville")
write_SQL(engine, conn, Bilbao_Weather_Data_df, "forecast_Bilbao")

## Read Data - SQLite
Madrid_df = pd.read_sql('select * from forecast_Madrid', conn)
Barcelona_df = pd.read_sql('select * from forecast_Barcelona', conn)
Valencia_df = pd.read_sql('select * from forecast_Valencia', conn)
Seville_df = pd.read_sql('select * from forecast_Seville', conn)
Bilbao_df = pd.read_sql('select * from forecast_Bilbao', conn)

model_test(Madrid_df, Barcelona_df)

# RBF - SVM Model fit best
city_compare(Madrid_df, Barcelona_df, "Barcelona")
city_compare(Madrid_df, Valencia_df, "Valencia")
city_compare(Madrid_df, Seville_df, "Seville")
city_compare(Madrid_df, Bilbao_df, "Bilbao")
city_compare(Madrid_df, Madrid_df, "Madrid")

sql_join(engine)