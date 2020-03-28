#This file include the preprocessing steps for Weather_city.csv; 
#Further used for Visualization
import pandas as pd
import matplotlib.pyplot as plt
# Input the five cities file, Set each city file with datetime index
Valencia_df=pd.read_csv('Resources/Valencia_Weather.csv',sep='\t')
Madrid_df=pd.read_csv('Resources/Madrid_Weather.csv',sep='\t')
Barcelona_df=pd.read_csv('Resources/Barcelona_Weather.csv',sep='\t')
Seville_df=pd.read_csv('Resources/Seville_Weather.csv',sep='\t')
Bilbao_df=pd.read_csv('Resources/Bilbao_Weather.csv',sep='\t')
Valencia_df=Valencia_df.set_index(['dt_iso'])
Madrid_df=Madrid_df.set_index(['dt_iso'])
Barcelona_df=Barcelona_df.set_index(['dt_iso'])
Seville_df=Seville_df.set_index(['dt_iso'])
Bilbao_df=Bilbao_df.set_index(['dt_iso'])
# Set up the time only from 2015 to 2018 
Madrid_df.index=pd.to_datetime(Madrid_df.index,utc=True)
Madrid_df=Madrid_df['2015':'2018']
Valencia_df.index=pd.to_datetime(Valencia_df.index,utc=True)
Valencia_df=Valencia_df['2015':'2018']
Barcelona_df.index=pd.to_datetime(Barcelona_df.index,utc=True)
Barcelona_df=Barcelona_df['2015':'2018']
Seville_df.index=pd.to_datetime(Seville_df.index,utc=True)
Seville_df=Seville_df['2015':'2018']
Bilbao_df.index=pd.to_datetime(Bilbao_df.index,utc=True)
Bilbao_df=Bilbao_df['2015':'2018']
#Delete all the duplicated data, and keep the first 
Madrid_df=Madrid_df.loc[~Madrid_df.index.duplicated(keep='first')]
Valencia_df=Valencia_df.loc[~Valencia_df.index.duplicated(keep='first')]
Barcelona_df=Barcelona_df.loc[~Barcelona_df.index.duplicated(keep='first')]
Seville_df=Seville_df.loc[~Seville_df.index.duplicated(keep='first')]
Bilbao_df=Bilbao_df.loc[~Bilbao_df.index.duplicated(keep='first')]
# Function creation; convert each time series data into the format of each day with 24 hours column[0~24]
# Isolate the value for target column
def hour_columns(data,target_col):
    data.loc[:,'year'] = data.index.year
    data.loc[:,'month'] = data.index.month
    data.loc[:,'day'] = data.index.day
    data.loc[:,'hours'] = data.index.hour
    data.loc[:,'date'] = pd.to_datetime(data.loc[:,['year', 'month', 'day']], format='%Y-%m-%d', errors='ignore')
    
    #set the index to dates only
    data = data.set_index(pd.DatetimeIndex(data['date']))
    #drop non target columns 
    data = data.loc[:,[target_col, 'hours']]
    #pivot the table into the format Date h0, h1, ...h23
    data = data.pivot(columns='hours', values=target_col)
    return data
# Create dataframe for each city, and looking for temperature column
Madrid=hour_columns(Madrid_df,'temp')
Valencia=hour_columns(Valencia_df,'temp')
Barcelona=hour_columns(Barcelona_df,'temp')
Seville=hour_columns(Seville_df,'temp')
Bilbao=hour_columns(Bilbao_df,'temp')

#Average of the temperature for each day; Create a function for average 
def total(data):
    data['Total']=data.mean(axis=1)
    return pd.DataFrame(data['Total'])
# input all the city dataframe into the average temperature function
new_Madrid=total(Madrid)
new_Valencia=total(Valencia)
new_Barcelona=total(Barcelona)
new_Seville=total(Seville)
new_Bilbao=total(Bilbao)

# Date conversion function, putting each month result into plots; the x and y label here is for Bilbao city
def datetime_conversion(data):
    data.index=data.index.map(lambda x:x.strftime('%Y-%m'))
    data.index.name='hours'
    data=data.mean(level='hours')
    data_2015=pd.DataFrame(data['Total'][0:12])
    data_2016=pd.DataFrame(data['Total'][12:24])
    data_2017=pd.DataFrame(data['Total'][24:36])
    data_2018=pd.DataFrame(data['Total'][36:48])
    month=['January','February','March','April','May','June','July','August','September','October','November','December']
    plt.plot(month,data_2015['Total'],label="2015",marker='.')
    plt.plot(month,data_2016['Total'],label="2016",marker='.')
    plt.plot(month,data_2017['Total'],label="2017",marker='.')
    plt.plot(month,data_2018['Total'],label="2018",marker='.')
    plt.xticks(rotation='vertical',fontsize=12)
    plt.xlabel("Month",fontsize=18)
    plt.ylabel("Avearage Temperature(Kelvin) ",fontsize=18)
    plt.title("Average Temperature of Bilbao(2015-2018)",fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    fig=plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig('Bilbao.png',dpi=100)
    plt.show()
    return data_2015, data_2016, data_2017,data_2018
    #Function for all five cities dataset, looking for the average temperature of 2018 for all 5 largest cities in spain

