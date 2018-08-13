import os
import sys

import numpy as np
import pandas as pd
import math

import collections as coll
from sklearn.neighbors import KDTree

import merge as merge

def assign_free_hits(labels, hits):
    """
    Assign singleton hits to small tracks and tracks with an even length.
    This does not lose majority count, while adding a small potential benefit.
    """
    def do_one_assignment_round(labels, hits, lengthen_short=False):
        labels = np.copy(labels)
        distances = np.zeros((len(labels)))
        unique_labels = np.unique(labels)
        label_counts = coll.Counter(labels).most_common(len(unique_labels))
        #count_free_hits = len(np.where(labels == 0)[0])
        #print('free hits available this round: ' + str(count_free_hits))

        hits['track_id'] = labels.tolist()

        df = hits.loc[(hits.track_id == 0)]
        hit_ids = df.hit_id.values

        gz_abs, gzr, grn, gc, gs = hits[['z_abs', 'zr', 'rn', 'c', 's']].values.T
        zr, rn, c, s = df[['zr', 'rn', 'c', 's']].values.T
        tree = KDTree(np.column_stack([c, s, rn, zr]), metric='euclidean')

        for label_count in label_counts:
            if (label_count[0] == 0): continue
            if (label_count[1] < 5) or ((not lengthen_short) and ((label_count[1] % 2) == 0)):
                idx = np.where(labels==label_count[0])[0]
                idx = idx[np.argsort(gz_abs[idx])]

                ## start and end points  ##
                idx0,idx1 = idx[0],idx[-1]
                gzr0 = gzr[idx0]
                gzr1 = gzr[idx1]
                grn0 = grn[idx0]
                grn1 = grn[idx1]
                gc0 = gc[idx0]
                gc1 = gc[idx1]
                gs0 = gs[idx0]
                gs1 = gs[idx1]

                (nearest_dist0, nearest_idx0) = tree.query([[gc0, gs0, grn0, gzr0]], k=1)
                nearest_dist0 = np.concatenate(nearest_dist0)
                nd0 = nearest_dist0[0]
                nearest_idx0 = np.concatenate(nearest_idx0)
                nidx0 = nearest_idx0[0]
                gidx0 = hit_ids[nidx0] - 1            

                (nearest_dist1, nearest_idx1) = tree.query([[gc1, gs1, grn1, gzr1]], k=1)
                nearest_dist1 = np.concatenate(nearest_dist1)
                nd1 = nearest_dist1[0]
                nearest_idx1 = np.concatenate(nearest_idx1)
                nidx1 = nearest_idx1[0]
                gidx1 = hit_ids[nidx1] - 1

                #print('end point 0: ' + str(gzr0))
                #print('free idx: ' + str(nidx0))
                #print('nearest point: ' + str(zr[nidx0]))
                #print('nearest distance: ' + str(nd0))
                #print('global index: ' + str(gidx0))
                #print('global zr: ' + str(gzr[gidx0]))

                if nd0 < nd1 and ((labels[gidx0] == 0) or (nd0 < distances[gidx0])):
                    labels[gidx0] = label_count[0]
                    distances[gidx0] = nd0
                elif ((labels[gidx1] == 0) or (nd1 < distances[gidx1])):
                    labels[gidx1] = label_count[0]
                    distances[gidx1] = nd1

        return labels

    labels = np.copy(labels)
    hits['z_abs'] = hits.z.abs()
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['rn'] = hits['r'] / 1000
    hits['a'] = np.arctan2(hits.y.values, hits.x.values)
    hits['c'] = np.cos(hits.a.values)
    hits['s'] = np.sin(hits.a.values)
    hits['zr'] = hits['z'] / hits['r']

    # First remove any single and double hit tracks
    (labels, _) = merge.remove_small_tracks(labels, smallest_track_size=3)
    count_free_hits = len(np.where(labels == 0)[0])
    #print('free hits available round 1a: ' + str(count_free_hits))

    if count_free_hits > 0:
        labels = do_one_assignment_round(labels, hits, lengthen_short=False)
        count_free_hits = len(np.where(labels == 0)[0])
        if count_free_hits > 0:
            labels = do_one_assignment_round(labels, hits)

    # Now remove any tracks with only three hits
    (labels, _) = merge.remove_small_tracks(labels, smallest_track_size=4)
    count_free_hits = len(np.where(labels == 0)[0])
    #print('free hits available at start of round 2: ' + str(count_free_hits))
    if count_free_hits > 0:
        labels = do_one_assignment_round(labels, hits, lengthen_short=False)
        count_free_hits = len(np.where(labels == 0)[0])
        if count_free_hits > 0:
            labels = do_one_assignment_round(labels, hits)

    return labels