import os
import sys

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import collections as coll

def get_truth_for_track(track, labels, truth):
    """Return the label indexes for the truth version of the given track."""
    hit_ix = np.where(labels == track)[0]
    tdf = truth.loc[hit_ix]
    truth_count = coll.Counter(tdf.particle_id.values).most_common(2)
    truth_particle_id = truth_count[0][0]
    if truth_particle_id == 0 and len(truth_count) > 1:
        truth_particle_id = truth_count[1][0]
    if truth_particle_id == 0:
        truth_ix = hit_ix
    else:
        tdf2 = truth.loc[(truth.particle_id == truth_particle_id)]
        tdf2 = tdf2.sort_values('tz')
        truth_ix = tdf2.index.values
    return truth_ix

def compare_track_to_truth(track, labels, hits, truth):
    """Display comparison results between a track and the ground truth track."""
    hit_ix = np.where(labels == track)[0]
    df = hits.loc[hit_ix]
    df = df.sort_values('z')
    dfx1 = df[['x','y', 'z', 'zr', 'volume_id', 'layer_id']]
    truth_ix = get_truth_for_track(track, labels, truth)

    arr_s1 = np.copy(hit_ix)
    arr_s1.sort()
    arr_s2 = np.copy(truth_ix)
    arr_s2.sort()

    print('Detected track: ' + str(arr_s1))
    print('Truth track:    ' + str(arr_s2))
    print(dfx1)
    if np.array_equal(arr_s1, arr_s2):
        print('Track equals truth')
    else:
        tdf = hits.loc[truth_ix]
        tdf = tdf.sort_values('z')
        tdf = tdf[['x','y', 'z', 'zr', 'volume_id', 'layer_id']]
        print(tdf)

def track_distance_from_truth(track, labels, hits, truth):
    """Return whether a track matches the ground truth exactly, or if not, the
    number of track hits that are correct truth track hits, vs incorrect (refer
    to another track)."""
    hit_ix = np.where(labels == track)[0]
    truth_ix = get_truth_for_track(track, labels, truth)

    arr_s1 = np.copy(hit_ix)
    arr_s1.sort()
    arr_s2 = np.copy(truth_ix)
    arr_s2.sort()

    correct = 0
    incorrect = 0
    is_match = np.array_equal(arr_s1, arr_s2)
    if not is_match:
        for hit in arr_s1:
            if hit in arr_s2:
                correct = correct + 1
            else:
                incorrect = incorrect + 1

    return (is_match, correct, incorrect)

def graph_my_track(track, labels, hits, truth=None):
    """Display xyz representation of given track in a graph."""
    hit_ix = np.where(labels == track)[0]
    df = hits.loc[hit_ix]
    df = df.sort_values('z')
    x,y,z = df[['x', 'y', 'z' ]].values.astype(np.float32).T
    track_dims = np.column_stack((x, y, z))
    if truth is not None:
        truth_ix = get_truth_for_track(track, labels, truth)
        tdf = hits.loc[truth_ix]
        tdf = tdf.sort_values('z')
        tx,ty,tz = tdf[['x', 'y', 'z' ]].values.astype(np.float32).T
        truth_dims = np.column_stack((tx, ty, tz))
    else:
        truth_dims = track_dims

    draw_prediction_xyz([truth_dims], [track_dims])

def draw_prediction_xyz(truth, predict):
    """Graph the given xyz values of a truth track and predicted track.
    The predicted track is in colour, the truth track is grey."""
    fig1 = plt.figure(figsize=(12,12))
    ax1  = fig1.add_subplot(111, projection='3d')
    fig1.patch.set_facecolor('white')
    ax1.set_xlabel('x', fontsize=16)
    ax1.set_ylabel('y', fontsize=16)
    ax1.set_zlabel('z', fontsize=16)

    predict_size = len(predict)
    #predict_size = 10
    for n in range(0,predict_size,1):
        x, y, z = predict[n].T
        ex, ey, ez = truth[n].T
        
        color = np.random.uniform(0,1,3)
        ax1.plot(ex,ey,ez,'.-',color = [0.75,0.75,0.75], markersize=10)
        ax1.plot(x,y,z,'.-',color = color, markersize=5)
        plt.axis('equal')
