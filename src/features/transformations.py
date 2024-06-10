import pandas as pd
import numpy as np
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date
from dateutil import parser


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:

    #Idea is to get the percentage of orders accepted out of all sent.
    
    #Selected only the ACCEPTED,REJECTED and IGNORED related rows
    df_processed = df[df['participant_status']!='CREATED'].reset_index(drop=True)
    
    #changing it to datetime
    df_processed['event_timestamp'] = df_processed['event_timestamp'].apply(lambda x: parser.parse(x))
    
    #performing cross join kind of operaton so the we will get all the orders of the driver and later we will compare the time of two orders.
    merged = pd.merge(df_processed, df_processed, on='driver_id', how='inner')
    
    #filtering only orders of older times to get the freq
    merged=merged[merged['event_timestamp_x']>merged['event_timestamp_y']].reset_index(drop=True)
    
    #for each order we will get the count of previously completed orderers and count of previously sent requests.
    merged_unique = merged.groupby(['event_timestamp_x','driver_id','participant_status_x','order_id_x','experiment_key_x',
                                    'driver_latitude_x','driver_longitude_x','driver_gps_accuracy_x','trip_distance_x',
                                    'pickup_latitude_x','pickup_longitude_x','is_completed_x'])['is_completed_y'].agg(['sum','count']).reset_index()
    
    #we will divide these to get the % of orders accepted out of requested    
    merged_unique['accept_freq'] = np.where(merged_unique['count']!=0,merged_unique['sum']/merged_unique['count'],0)
    
    #selected the only required columns to merge it to the original dataset
    merged_unique = merged_unique[['driver_id','order_id_x','accept_freq']]
    merged_unique.rename(columns = {'order_id_x':'order_id'}, inplace = True)
                                    
    #merging the calculated and original dataframe.                            
    df_final = df.merge(merged_unique,on=['driver_id','order_id'],how='left')
    
    #for the first ordere we will populated with 40%, as we don't have enough data to tell during first time so assuming the chance to be less than 50%
    df_final['accept_freq'] = df_final['accept_freq'].fillna(0.4)
    return df_final
    
    '''
    raise NotImplementedError(
        f"Show us your feature engineering skills! Suppose that drivers with a good track record are more likely to accept bookings. "
        f"Implement a feature that describes the number of historical bookings that each driver has completed."
    )
    '''