import pandas as pd
import numpy as np
from AntColonyOptimization import ACO, Graph
from plot import plot
import datetime
import pickle
import argparse
import xgboost as xgb
import pprint
import matplotlib.pyplot as plt


loaded_model = pickle.load(open("xgb_model.sav", 'rb'))


def time_cost_between_points(point1, point2, passenger_count, store_and_fwd_flag=0):
    """
    Calculate the time between two points
    using the already trained XGB model
    """
    # As the dataset is of 2016 so for understandable results constant date
    date_list = [27, 5, 2016]

    year, month, day = int(date_list[2]), int(date_list[1]), int(date_list[0])

    my_date = datetime.date(year, month, day)

    model_data = {'passenger_count': passenger_count,
                  'pickup_longitude': point1['x'],
                  'pickup_latitude': point1['y'],
                  'dropoff_longitude': point2['x'],
                  'dropoff_latitude': point2['y'],
                  'store_and_fwd_flag': bool(store_and_fwd_flag),
                  'pickup_month': my_date.month,
                  'pickup_day': my_date.day,
                  'pickup_weekday': my_date.weekday(),
                  'pickup_hour': 23,
                  'pickup_minute': 10,
                  'latitude_difference': point2['y'] - point1['y'],
                  'longitude_difference': point2['x'] - point1['x'],
                  'trip_distance': trip_distance_cost(point1, point2)
                  }

    df = pd.DataFrame([model_data], columns=model_data.keys())
    pred = np.exp(loaded_model.predict(xgb.DMatrix(df))) - 1
    return pred[0]


def trip_distance_cost(point1, point2):
    """
    Calculate the manhattan distance between two points.
    """
    return 0.621371 * 6371 * (
        abs(2 * np.arctan2(np.sqrt(np.square(
            np.sin((abs(point2['y'] - point1['y']) * np.pi / 180) / 2))),
            np.sqrt(1-(np.square(np.sin((abs(point2['y'] - point1['y']) * np.pi / 180) / 2)))))) +
        abs(2 * np.arctan2(np.sqrt(np.square(np.sin((abs(point2['x'] - point1['x']) * np.pi / 180) / 2))),
                           np.sqrt(1-(np.square(np.sin((abs(point2['x'] - point1['x']) * np.pi / 180) / 2)))))))


# -----------------------------------------------------------------------------------------
# Command Line Args
parser = argparse.ArgumentParser()
parser.add_argument("loc_count", type=int)
parser.add_argument("ant_count", type=int,)
parser.add_argument("g", type=int)
parser.add_argument("alpha", type=float,)
parser.add_argument("beta", type=float)
parser.add_argument("rho", type=float)
parser.add_argument("q", type=float)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


locations = []
points = []

# Read the amount of data that is given by user

df = pd.read_csv("NewYork_Taxi_Data_Kaggle/test.csv")[:args.loc_count]
for index, row in df.iterrows():
    locations.append({
        'index': index,
        'x': row['pickup_longitude'],
        'y': row['pickup_latitude']
    })
    points.append((row['pickup_longitude'], row['pickup_latitude']))


# Build complete cost matrix based on time between points
cost_matrix = []
rank = len(locations)
for i in range(rank):
    row = []
    for j in range(rank):
        row.append(time_cost_between_points(locations[i], locations[j], 1, 0))
    cost_matrix.append(row)


aco = ACO(ant_count=args.ant_count, generations=args.g, alpha=args.alpha,
          beta=args.beta, rho=args.rho, q=args.q)

# graph with predicted distance as edge and number of points
graph = Graph(cost_matrix, rank)

best_path, cost, avg_costs, best_costs = aco.solve(graph, args.show)
print('====================================================================')
print('Final cost: {} minutes, path: {}'.format(cost, best_path))


plot(points, best_path)

x_values = [i for i in range(args.g)]
plt.title("Best Cost vs Generation for " + str(args.ant_count) + " Ants")
plt.ylabel("Best Cost")
plt.xlabel("Current Generation")
plt.plot(x_values, best_costs)
plt.show()

x_values = [i for i in range(args.g)]
plt.title("Avg Cost vs Generation for " + str(args.ant_count) + " Ants")
plt.ylabel("Avg Cost")
plt.xlabel("Current Generation")
plt.plot(x_values, avg_costs)
plt.show()
