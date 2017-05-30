# This script is a compilation of all functions needed for data preprocessing, cleaning and feature engineering
# Modified and added feature date --> month, year, day; added lagged feature (1,3,5,8,12) from weather data
# Fixed leakage in data, added closest station to train and test files
# Created dummies of species 

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


def read_input_files(dir_path):
    '''
    Read the input files: train.csv, test.csv, sampleSubmission.csv, spray.csv, weather.csv, mapdata.txt
    '''
    train    = pd.read_csv('%s/train.csv' % dir_path)
    test     = pd.read_csv('%s/test.csv' % dir_path)
    sample = pd.read_csv('%s/sampleSubmission.csv' % dir_path)
    spray    = pd.read_csv('%s/spray.csv' % dir_path)
    weather  = pd.read_csv('%s/weather.csv' % dir_path)
    mapdata  = np.loadtxt('%s/mapdata_copyright_openstreetmap_contributors.txt' %dir_path)
    
    return train, test, spray, weather, mapdata
    

def get_datetime(date_string, possible_fmts=['%Y-%m-%d', '%d/%m/%Y', '%d/%m/%y', '%Y%m%d']): 
    '''
    Change date to datetime function
    '''
    if pd.isnull(date_string):
        return date_string
    for fmt in possible_fmts:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
            
    return np.NaN
    
def get_lagged_day_feature(df, lag):
    '''
    lag - list of numbers defining lagged values. Builds lagged weather features
    '''
    new_dict={}
    for col_name in df:
        new_dict[col_name]=df[col_name]
        # create lagged Series
        for l in lag:
            if col_name!='Date' and col_name!='Station':
                new_dict['%s_lag%d' %(col_name,l)]=df[col_name].shift(l)
    res=pd.DataFrame(new_dict,index=df.index)
    
    return res
    
def get_dummies(df):   
    '''
    Constructs dummy indicators for species
    '''
    dummies=pd.get_dummies(df['Species'])
    df = pd.concat([df, dummies], axis=1)
    return df 
    
def process_date(data):
    '''
    Extract the year, month and day from the date
    '''
    data['year' ] = data['Date'].apply(lambda x : get_datetime(x).year)
    data['month'] = data['Date'].apply(lambda x : get_datetime(x).month)
    data['day'  ] = data['Date'].apply(lambda x : get_datetime(x).day)
    
    return data
    
def process_weather(weather):  
    '''
    Construct lagged weather values used as features. Replace "Trace" with 0.001, replace M and missing with NaN.
    Replace missing WetBulb of 1st station with the value of 2d station. Replace all missing values of 2d station with values of 1st station
    Build lagged features for 1st station and for 2d station.
    '''
    days = [1, 3, 5, 8, 12]     
    weather.sort(['Date', 'Station'], axis=0, ascending=True, inplace=True)
    drop_cols = ['Heat', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'StnPressure',  'SeaLevel', 'AvgSpeed' ]
    weather.drop(drop_cols, axis=1, inplace=True) 
    weather.replace(['  T','M','-'], [0.001, np.nan, np.nan], inplace=True) 
    weather.WetBulb.fillna(method='bfill', inplace=True)   
    weather.fillna(method='pad', inplace=True)
    weather1 = get_lagged_day_feature(weather[weather['Station']==1], days) 
    weather2 = get_lagged_day_feature(weather[weather['Station']==2], days) 
    weather = weather1.append(weather2)                                  
    weather.sort(['Date', 'Station'], axis=0, ascending=True, inplace=True)
    
    return weather


def get_closest_station(df):
    '''
    Identify the closest weather station
    '''
    df['lat1']=41.995   # latitude of 1st station
    df['lat2']=41.786   # latitude of 2d station
    df['lon1']=-87.933  # longitude of 1st station
    df['lon2']=-87.752  # longitude of 2d station
    df['dist1'] = haversine(df.Latitude.values, df.Longitude.values, df.lat1.values, df.lon1.values) #calculate distance
    df['dist2'] = haversine(df.Latitude.values, df.Longitude.values, df.lat2.values, df.lon2.values)
    indicator = np.less_equal(df.dist1.values, df.dist2.values) # determine which station is the closest
    st = np.ones(df.shape[0])
    st[indicator==0]=2
    df['Station']=st   
    df.drop(['dist1', 'dist2', 'lat1', 'lat2', 'lon1', 'lon2' ], axis=1, inplace=True)
    
    return df


def get_duplicated_rows(df): 
    '''
    Calculates number of duplicated rows by Date, Trap, Species
    '''
    grouped = df.groupby(['Date', 'Trap', 'Species'])
    num=grouped.count().Latitude.to_dict()
    df['N_Dupl']=-999
    for idx in df.index:
        d = df.loc[idx, 'Date']
        t = df.loc[idx, 'Trap']
        s = df.loc[idx, 'Species']
        df.loc[idx, 'N_Dupl'] = num[(d, t, s)]
        
    return df
    

def merge_weather(df1, df2):
    result = df1.merge(df2, on=['Date', 'Station'], how="left",  left_index=True)
    
    return result 


def fix_leakage(df):  
    '''
    Assigns WNV indicator to duplicate rows (if WNV=1, assign 1 to all 
    duplicate rows)
    '''
    grouped = df.groupby(by=['Date', 'Trap', 'Species'], as_index=False)['WnvPresent'].max() 
    df.drop('WnvPresent', axis=1, inplace=True)
    grouped.columns = ['Date', 'Trap', 'Species', 'WnvPresent']
    result = df.merge(grouped, on=['Date', 'Trap', 'Species'], how="left") #.reset_index()
    
    return result


def haversine(lat1, lon1, lat2, lon2): 
    '''
    Calculates the haversine distance between two Lat, Long pairs
    '''
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a    = np.power(np.sin(dLat/2), 2) + np.multiply(np.cos(lat1), np.multiply(np.cos(lat2), np.power(np.sin(dLon/2), 2)))
    c    = 2*np.arcsin(np.sqrt(a))
    
    return R * c
    

def prepare_data_for_model(train, test):
    '''
    log(# of mosquitos) is a target variable in stacking procedure
    prediction target for classification model
    '''
    y_reg   = np.log(train.NumMosquitos.values+1) 
    y_clf = train.WnvPresent.values    
    train.drop(['NumMosquitos', 'WnvPresent'], axis=1, inplace=True)
    ids=test.Id.values    
    test.drop('Id', axis=1, inplace=True)
    for c in train.columns:
        if train[c].dtype=='object' :
            train[c] = train[c].astype(float)
            test[c]  = test[c].astype(float)
            
    return train, test, y_reg, y_clf, ids

    



