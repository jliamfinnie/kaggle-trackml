import numpy as np
import pandas as pd
import math
import collections as coll
import zroutlier as zro

def non_outlier_len(track_id, labels, hits, outlier_ix):
    """
    Determine the current length of the track, as well as projected length of the track
    if any identified outliers are removed. contains_outlier is set depending on whether
    the provided outlier_ix is found in the list of potential track outliers.
    """
    labels = np.copy(labels)
    orig_len = len(np.where(labels == track_id)[0])
    contains_outlier = False
    outliers = zro.find_track_outliers_zr(track_id, labels, hits, find_all=True)
    if not contains_outlier:
        contains_outlier = (outlier_ix in outliers)
    for outx in outliers:
        labels[outx] = 0
    if len(outliers) > 0:
        outliers = zro.find_track_outliers_zr(track_id, labels, hits, find_all=True)
        if not contains_outlier:
            contains_outlier = (outlier_ix in outliers)
        for outx in outliers:
            labels[outx] = 0
        if len(outliers) > 0:
            outliers = zro.find_track_outliers_zr(track_id, labels, hits, find_all=True)
            if not contains_outlier:
                contains_outlier = (outlier_ix in outliers)
            for outx in outliers:
                labels[outx] = 0
            if len(outliers) > 0:
                outliers = zro.find_track_outliers_zr(track_id, labels, hits, find_all=True)
                if not contains_outlier:
                    contains_outlier = (outlier_ix in outliers)
                for outx in outliers:
                    labels[outx] = 0
    new_len = len(np.where(labels == track_id)[0])
    return (orig_len, new_len, contains_outlier)

def calculate_track_score(track_id, labels, hits, outlier_modifier=0.75, outlier_ix=-1):
    """
    Calculate an approximate track score from 0.0 to 1.0, where 1.0 is a high-quality
    long track. If the provided outlier_ix is found to be an actual outlier, the resulting
    score will be multiplied by the specified outlier_modifier.
    """
    (cur_len, no_outlier_len, has_outlier) = non_outlier_len(track_id, labels, hits, outlier_ix)
    modifier=1.0
    if has_outlier:
        modifier=outlier_modifier
    score1 = no_outlier_len / cur_len
    if cur_len < 4:
        score2 = 0
    else:
        score2 = min(cur_len/20.0, 1.0)
    return ((score1 + score2) * modifier)/2.0

