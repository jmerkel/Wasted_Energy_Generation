#5 largest city in Spain average temperature plot creation Function;
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
def datetime_conversion(data_1,data_2,data_3,data_4,data_5):
    data_1.index=data_1.index.map(lambda x:x.strftime('%Y-%m'))
    data_2.index=data_2.index.map(lambda x:x.strftime('%Y-%m'))
    data_3.index=data_3.index.map(lambda x:x.strftime('%Y-%m'))
    data_4.index=data_4.index.map(lambda x:x.strftime('%Y-%m'))
    data_5.index=data_5.index.map(lambda x:x.strftime('%Y-%m'))
    data_1.index.name='hours'
    data_2.index.name='hours'
    data_3.index.name='hours'
    data_4.index.name='hours'
    data_5.index.name='hours'
    data_1=data_1.mean(level='hours')
    data_2=data_2.mean(level='hours')
    data_3=data_3.mean(level='hours')
    data_4=data_4.mean(level='hours')
    data_5=data_5.mean(level='hours')
    data_2018_1=pd.DataFrame(data_1['Total'][36:48])
    data_2018_2=pd.DataFrame(data_2['Total'][36:48])
    data_2018_3=pd.DataFrame(data_3['Total'][36:48])
    data_2018_4=pd.DataFrame(data_4['Total'][36:48])
    data_2018_5=pd.DataFrame(data_5['Total'][36:48])
    month=['January','February','March','April','May','June','July','August','September','October','November','December']
    plt.plot(month,data_2018_1['Total'],label="Madrid",marker='.')
    plt.plot(month,data_2018_2['Total'],label="Valencia",marker='.')
    plt.plot(month,data_2018_3['Total'],label="Barcelona",marker='.')
    plt.plot(month,data_2018_4['Total'],label="Seville",marker='.')
    plt.plot(month,data_2018_5['Total'],label="Bilbao",marker='.')
    plt.xticks(rotation='vertical',fontsize=12)
    plt.xlabel("Month",fontsize=18)
    plt.ylabel("Avearage Temperature(Kelvin) ",fontsize=18)
    plt.title("Average Temperature of 5 Cities(2018)",fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    fig=plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig('5Cities_2018.png',dpi=100)
    plt.show()
    return data_2018_1,data_2018_2,data_2018_3,data_2018_4,data_2018_5
datetime_conversion(new_Madrid,new_Valencia,new_Barcelona,new_Seville,new_Bilbao)