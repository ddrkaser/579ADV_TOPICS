import numpy as np
#import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
import sys
import pandas as pd


walking = pd.read_json('walking.json')
running = pd.read_json('running.json')
jumping = pd.read_json('jumping.json')

de = walking.describe()
walking.info()
walking.columns
#cols = ['sensor', 'time', 'seconds_elapsed', 'z', 'y', 'x']
cols = ['z', 'y', 'x']

walking['sensor'].value_counts()

#Orientation doesn't matter in activity
walking = walking[walking['sensor'].isin(['Accelerometer','Gravity','Gyroscope'])]
walking_acc = walking[walking['sensor'] == 'Accelerometer'][cols]
walking_acc.columns = ['acc_z', 'acc_y', 'acc_x']
walking_gra = walking[walking['sensor'] == 'Gravity'][cols]
walking_gra.columns = ['gra_z', 'gra_y', 'gra_x']
walking_gyr = walking[walking['sensor'] == 'Gyroscope'][cols]
walking_gyr.columns = ['gyr_z', 'gyr_y', 'gyr_x']

walking_n = pd.concat([walking_acc.reset_index(drop=True), walking_gra.reset_index(drop=True), walking_gyr.reset_index(drop=True)], axis = 1)

walking_n = pd.merge([walking_acc, walking_gra, walking_gyr],)
walking.info()
walking[cols]

def pre_process(df):
    #df = df[df['sensor'].isin(['Accelerometer','Gravity','Gyroscope'])]
    cols = ['z', 'y', 'x']
    df_acc = df[df['sensor'] == 'Accelerometer'][cols]
    df_acc.columns = ['acc_z', 'acc_y', 'acc_x']
    df_gra = df[df['sensor'] == 'Gravity'][cols]
    df_gra.columns = ['gra_z', 'gra_y', 'gra_x']
    df_gyr = df[df['sensor'] == 'Gyroscope'][cols]
    df_gyr.columns = ['gyr_z', 'gyr_y', 'gyr_x']
    df_n = pd.concat([df_acc.reset_index(drop=True),df_gra.reset_index(drop=True), df_gyr.reset_index(drop=True)], axis = 1)
    return df_n

walking = pre_process(walking)
running = pre_process(running)
jumping = pre_process(jumping)

def find_clusters(dist_ma,linkage = None):
    clusters = {}
    row_index = -1
    col_index = -1
    array = []

    for n in range(dist_ma.shape[0]):
        array.append(n)

    clusters[0] = array.copy()

    #finding minimum value from the distance matrix
    #note that this loop will always return minimum value from bottom triangle of matrix
    for k in range(1, dist_ma.shape[0]):
        min_val = sys.maxsize

        for i in range(0, dist_ma.shape[0]):
            for j in range(0, dist_ma.shape[1]):
                if(dist_ma[i][j]<=min_val):
                    min_val = dist_ma[i][j]
                    row_index = i
                    col_index = j

        #once we find the minimum value, we need to update the distance matrix
        #updating the matrix by calculating the new distances from the cluster to all points

        #for Complete Linkage
        if(linkage=='farthest'):
             for i in range(0,dist_ma.shape[0]):
                if(i != col_index and i!=row_index):
                    temp = min(dist_ma[col_index][i],dist_ma[row_index][i])
                    dist_ma[col_index][i] = temp
                    dist_ma[i][col_index] = temp
        #for Average Linkage
        else:
             for i in range(0,dist_ma.shape[0]):
                if(i != col_index and i!=row_index):
                    temp = (dist_ma[col_index][i]+dist_ma[row_index][i])/2
                    dist_ma[col_index][i] = temp
                    dist_ma[i][col_index] = temp

        #set the rows and columns for the cluster with higher index i.e. the row index to infinity
        #Set input[row_index][for_all_i] = infinity
        #set input[for_all_i][row_index] = infinity
        for i in range (0,dist_ma.shape[0]):
            dist_ma[row_index][i] = sys.maxsize
            dist_ma[i][row_index] = sys.maxsize

        #Manipulating the dictionary to keep track of cluster formation in each step
        #if k=0,then all datapoints are clusters

        minimum = min(row_index,col_index)
        maximum = max(row_index,col_index)
        for n in range(len(array)):
            if(array[n]==maximum):
                array[n] = minimum
        clusters[k] = array.copy()

    return clusters

tt = running.iloc[100:200,].reset_index(drop=True).to_numpy()
initial_distances = pairwise_distances(tt,metric='euclidean')
#making all the diagonal elements infinity
np.fill_diagonal(initial_distances,sys.maxsize)
clusters = find_clusters(initial_distances)


