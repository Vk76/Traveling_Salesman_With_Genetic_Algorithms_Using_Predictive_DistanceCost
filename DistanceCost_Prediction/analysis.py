import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def clean(data):

    duration_mask = ((data.trip_duration < 60) |
                     (data.trip_duration > 3600 * 2))

    print('Anomalies in trip duration, %: {:.2f}'.format(
        data[duration_mask].shape[0] / data.shape[0] * 100))

    data = data[~duration_mask]
    data.trip_duration = data.trip_duration.astype(np.uint16)
    print('Trip duration in seconds: {} to {}'.format(
        data.trip_duration.min(), data.trip_duration.max()))

    print('Empty trips: {}'.format(data[data.passenger_count == 0].shape[0]))
    data = data[data.passenger_count > 0]

    # Convert this feature into categorical type
    data.store_and_fwd_flag = data.store_and_fwd_flag.astype('category')

    # month (pickup and dropoff)
    data['mm_pickup'] = data.pickup_datetime.dt.month.astype(np.uint8)
    data['mm_dropoff'] = data.dropoff_datetime.dt.month.astype(np.uint8)
    # day of week
    data['dow_pickup'] = data.pickup_datetime.dt.weekday.astype(np.uint8)
    data['dow_dropoff'] = data.dropoff_datetime.dt.weekday.astype(np.uint8)

    # day hour
    data['hh_pickup'] = data.pickup_datetime.dt.hour.astype(np.uint8)
    data['hh_dropoff'] = data.dropoff_datetime.dt.hour.astype(np.uint8)

    return data


def PickupPlot1(data):
    '''
    Pickup time distribution by hour-of-day
    '''
    print(data)
    plt.figure(figsize=(12, 2))

    data = data.groupby('hh_pickup').aggregate({'id': 'count'}).reset_index()
    sns.barplot(x='hh_pickup', y='id', data=data)

    plt.title('Pick-ups Hour Distribution')
    plt.xlabel('Hour of Day, 0-23')
    plt.ylabel('No of Trips made')


def PickupPlot2(data, dow_names):
    '''
    Pickup time distribution by day-of-week
    '''
    plt.figure(figsize=(12, 2))

    data = data.groupby('dow_pickup').aggregate({'id': 'count'}).reset_index()
    sns.barplot(x='dow_pickup', y='id', data=data)

    plt.title('Pick-ups Weekday Distribution')
    plt.xlabel('Trip Duration, minutes')
    plt.xticks(range(0, 7), dow_names, rotation='horizontal')
    plt.ylabel('No of Trips made')


def PickupPlot3(data, dow_names):
    '''
    Pickup heatmap of day-of-week vs. hour-of-day
    '''
    plt.figure(figsize=(12, 2))
    sns.heatmap(data=pd.crosstab(data.dow_pickup, data.hh_pickup,
                                 values=data.vendor_id, aggfunc='count', normalize='index'))

    plt.title('Pickup heatmap, Day-of-Week vs. Day Hour')
    plt.ylabel('Weekday')
    plt.xlabel('Day Hour, 0-23')
    plt.yticks(range(0, 7), dow_names[::-1], rotation='horizontal')


def TripDurationPlot1(data):
    '''
    Trip duration distribution in minutes
    '''
    plt.figure(figsize=(12, 3))

    plt.title('Trip Duration Distribution')
    plt.xlabel('Trip Duration, minutes')
    plt.ylabel('No of Trips made')
    plt.hist(data.trip_duration / 60, bins=100)


def TripDurationPlot2(data, dow_names):
    '''
    Trip duration based on hour-of-day vs. weekday
    '''
    plt.figure(figsize=(12, 2))
    sns.heatmap(data=pd.crosstab(data.dow_pickup, data.hh_pickup,
                                 values=data.trip_duration/60, aggfunc='mean'))

    plt.title('Trip duration heatmap (Minutes), Day-of-Week vs. Day Hour')
    plt.ylabel('Weekday')
    plt.xlabel('Day Hour, 0-23')
    plt.yticks(range(0, 7), dow_names[::-1], rotation='horizontal')


if __name__ == '__main__':

    taxiDB = pd.read_csv(filepath_or_buffer='../NewYork_Taxi_Data_Kaggle/train.csv',
                         engine='c', infer_datetime_format=True, parse_dates=[2, 3])

    taxiDB = clean(taxiDB)

    dow_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']

    PickupPlot1(taxiDB)
    PickupPlot2(taxiDB, dow_names)
    PickupPlot3(taxiDB, dow_names)

    TripDurationPlot1(taxiDB)
    TripDurationPlot2(taxiDB, dow_names)
