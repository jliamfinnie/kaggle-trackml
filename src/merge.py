import numpy as np
import pandas as pd
import math
import collections as coll
import zroutlier as zro

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def renumber_labels(labels):
    new_label = 0
    for old_label in np.unique(labels):
        if not old_label == new_label:
            labels[labels == old_label] = new_label
        new_label += 1

    return labels

def heuristic_merge_tracks(labels1, labels2, hits, overwrite_limit=4, weak_tracks=False, print_summary=False):
    """ Merge tracks from two arrays of track labels.

    Merges are handled as follows:
     - tracks from labels2 are identified and searched
     - for each track from labels2:
       - use track directly if no conflict with any tracks from labels1
       - skip if labels1 already contains the same track of equal (or longer) length
       - otherwise, if there are potentially multiple conflicting tracks from labels1
         - if labels1 only contains a single track ID, as well as un-classified (0) hits,
           re-assign '0' track ID to labels1 track ID (i.e. lengthen the track)
         - otherwise, labels1 contains multiple non-zero track IDs
           - replace any track ID 0 occurrences with the longest labels1 track ID
           - for short tracks, evaluate whether hits from those tracks should belong
             to the existing track, or the new track, based on outlier evaluations.

    Parameters:
     - labels1: np array of labels, each entry represents a hit, the value represents the
       track ID that hit is assigned to. This should be considered the 'higher-quality' of
       the two input labels
     - labels2: np array of secondary labels, whose tracks should be merged into labels1
     - hits: Hit information, including co-ordinates, volume/layer IDs, z/r values, etc.
     - overwrite_limit: Tracks longer than this limit are less likely to be re-assigned to
       a new track.
     - weak_tracks: If labels2 are assumed to be tracks of weak quality, the merging
       heuristics are made more conservative, such that existing labels1 tracks are less
       likely to be re-assigned to tracks from labels2.

    Returns: The merged array of labeled tracks.
    """
    labels_merged = np.copy(labels1)
    labels_merged = renumber_labels(labels_merged)
    max_track = np.amax(labels_merged)
    labels2[labels2 != 0] = labels2[labels2 != 0] + max_track
    trks2 = np.unique(labels2)
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count4_len = []
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    count10 = 0
    count11 = 0
    count12 = 0
    count13 = 0
    count14 = 0
    count15 = 0
    count16 = 0
    for trk2 in trks2:
        if trk2 == 0:
            continue
        trk2_ix = np.where(labels2 == trk2)[0]
        trk2_length = len(trk2_ix)
        if trk2_length < 2:
            continue
        trk1_val = labels_merged[trk2_ix]
        #print('trk2: ' + str(trk2) + ', label1: ' + str(trk1_val))
        trk1_uniq = np.unique(trk1_val)
        # Now we know which tracks from the 1st label overlap with the tracks from the 2nd label
        if len(trk1_uniq) == 1:
            if trk1_uniq[0] == 0:
                #print('Good candidate to replace!')
                # This track was not found by labels1, just directly use the
                # track from labels2.
                count1 = count1 + 1
                labels_merged[trk2_ix] = trk2
            else:
                # We found a track that is at least as long as the current
                # track in labels1. Nothing more needed, at least for now.
                # We could consider scenarios where the labels1 track contains
                # hits from 2 different tracks, where labels2 only has a
                # shorter single track. In this case, it may be good to split
                # the labels1 track into two pieces. However, this condition
                # would be very hard to detect, for now we want to favour
                # longer tracks whenever possible.
                #print('Same track found, skipping...')
                count2 = count2 + 1
        else:
            found_tracks = 0
            # Get counts for all identified tracks from labels1 that match trk2
            trk1_counts = coll.Counter(trk1_val).most_common(len(trk1_uniq))
            longest_track_id = trk1_counts[0][0]
            longest_track_count = trk1_counts[0][1]
            second_track_id = trk1_counts[1][0]
            second_track_count = trk1_counts[1][1]
            # If longest track in labels1 was 0, create a new track, but only
            # from free hits, or from small tracks. Also, if there is not
            # enough overlap (less than half the hits overlap), also create
            # a new track.
            if longest_track_id == 0:
                count5 = count5 + 1
                longest_track_id = trk2
                # See if we should instead lengthen the longest non-zero track
                if len(trk1_uniq) == 2:
                    test_track_ix = np.where(labels_merged == second_track_id)[0]
                    if len(test_track_ix) <= (second_track_count + 3):
                        outliers1 = zro.find_track_outliers_zr(second_track_id, labels_merged, hits, find_all=True)
                        labelx = np.copy(labels_merged)
                        labelx[trk2_ix] = second_track_id
                        outliers2 = zro.find_track_outliers_zr(second_track_id, labelx, hits, find_all=True)
                        if len(outliers2) <= len(outliers1):
                            longest_track_id = second_track_id
                            longest_track_count = second_track_count
            elif (trk2_length > 20) or (longest_track_count > 20):
                count9 = count9 + 1
                longest_track_id = trk2
            elif (trk2_length > 6) and (longest_track_count < int(trk2_length/2)) and second_track_id != 0:
                # Try to avoid creating crossed tracks, do not lengthen existing track if not
                # enough overlap.
                count10 = count10 + 1
                trk1a = np.where(labels_merged == longest_track_id)[0]
                if longest_track_count + 3 >= len(trk1a):
                    #print('Top 2 tracks, new: ' + str(trk2_length) + ', len1: '  + str(longest_track_count) + ', len1a: ' + str(len(trk1a)) + ', len2: ' + str(second_track_count) + ', len2a: ' + str(len(trk2a)))
                    # Lengthen the longest track, it's fully contained by our new/proposed track.
                    # Reset 2nd longest track if mostly contained in new/proposed track.
                    count14 = count14 + 1
                    trk2a = np.where(labels_merged == second_track_id)[0]
                    if second_track_count + 1 >= len(trk2a):
                        count15 = count15 + 1
                        labels_merged[trk2a] = longest_track_id
                else:
                    # Not much overlap, start a new track to avoid hurting existing tracks.
                    longest_track_id = trk2
            else:
                # If the old track had too many hits not part of the new/proposed track, do
                # not lengthen it - that may lose majority. Better to start a new track.
                trk1a = np.where(labels_merged == longest_track_id)[0]
                if weak_tracks:
                    if longest_track_count + 3 < len(trk1a):
                        count16 = count16 + 1
                        longest_track_id = trk2
                elif longest_track_count + 6 < len(trk1a):
                    count16 = count16 + 1
                    longest_track_id = trk2

            for trk1 in trk1_uniq:
                if trk1 == 0:
                    continue
                trk1_ix = np.where(labels_merged == trk1)[0]
                if len(trk1_ix) > 1:
                    found_tracks = found_tracks + 1
            if found_tracks > 1:
                #print('Found ' + str(found_tracks) + ' non-trivial tracks.')
                count3 = count3 + 1
                # If there are un-classified hits, assign those to the track
                # ID with the most hits.
                for label_ix in trk2_ix:
                    if labels_merged[label_ix] == 0:
                        labels_merged[label_ix] = longest_track_id
                        count6 = count6 + 1

                # If there are tracks of length 2 or less, and one or both
                # of those hits are included in the target track, re-assign
                # those matching the labels2 track to the most common
                # original track ID.
                for trk1_count in trk1_counts:
                    trk1_count_ix = np.where(labels_merged == trk1_count[0])[0]
                    if len(trk1_count_ix) <= overwrite_limit:
                        outliers = zro.find_track_outliers_zr(trk2, labels2, hits, find_all=True)
                        for label_ix in trk2_ix:
                            if labels_merged[label_ix] == trk1_count[0] and label_ix in outliers:
                                count13 = count13 + 1
                            elif labels_merged[label_ix] == trk1_count[0]:# and label_ix not in outliers:
                                labels_merged[label_ix] = longest_track_id
                                count7 = count7 + 1
                    elif not weak_tracks:
                        outliers = zro.find_track_outliers_zr(trk1_count[0], labels_merged, hits, find_all=True)
                        for label_ix in trk2_ix:
                            if labels_merged[label_ix] == trk1_count[0] and label_ix in outliers:
                                labels_merged[label_ix] = longest_track_id
                                count12 = count12 + 1

            else:
                # Only the track ID, as well as track ID 0, were found in labels1.
                # Replace any occurrences of ID 0 with the labels1 track ID.
                count4 = count4 + 1
                count4_len.append(len(trk2_ix))
                # If there are un-classified hits, assign those to the track
                # ID with the most hits (lengthens the track).
                for label_ix in trk2_ix:
                    if labels_merged[label_ix] == 0:
                        labels_merged[label_ix] = longest_track_id
                        count8 = count8 + 1

    if print_summary:
        print('Simple replacement of unclassified hits: ' + str(count1))
        print('Similar tracks (no-op): ' + str(count2))
        print('New track creations from little overlap(0): ' + str(count5))
        print('New track creations from huge tracks(>20): ' + str(count9))
        print('Test for new track creations from little overlap(non-0): ' + str(count10))
        print('--> Lengthen longest overlap instead: ' + str(count14))
        print('  --> And clear 2nd longest track: ' + str(count15))
        print('Skip extension due to too little overlap: ' + str(count16))
        print('Multiple non-trivial tracks: ' + str(count3))
        print('--> of which partial track ID 0 hits were updated: ' + str(count6))
        print('--> of which partial track ID non-0 hits were updated: ' + str(count7))
        print('--> of which partial track ID non-0 hits were skipped: ' + str(count13))
        print('--> of which outliers were overwritten: ' + str(count12))
        print('Tracks to be lengthened: ' + str(count4))
        print('--> of which track ID 0 hits were updated: ' + str(count8))
        print('--> from which new tracks were created instead: ' + str(count11))
        noises = np.unique(np.asarray(count4_len))
        print('--> of which labels2 unique track lengths were: ' + str(noises))

    return labels_merged


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_furthest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmax()
    return array[idx]

def find_dupzr_outlier(zr1, zr2, zrdup1, zrdup2):
    diff_zr = zr2 - zr1
    new_zr = zr2 + diff_zr
    outlier_zr = find_furthest_value([zrdup1, zrdup2], new_zr)
    return outlier_zr

def all_zs_unique(zs, num_hits_in_layer, source_ix):
    uniq = np.unique(zs[source_ix:source_ix+num_hits_in_layer])
    return len(uniq) == num_hits_in_layer

def can_detect_dupzr(old_zr, new_zr, zs, num_hits_in_layer, source_ix):
    possible = (old_zr != 0 and new_zr != 0)
    if not possible:
        mc = num_hits_in_layer
        for ii in range(mc):
            if (ii % 2 == 0) and ii < (mc - 1) and zs[source_ix+ii] != zs[source_ix+ii+1]:
                possible = True
    return possible

def find_dupzr_in_layer(old_zr, new_zr, zs, zrs, num_hits_in_layer, source_ix):
    mc = num_hits_in_layer
    duplicatez_source_ix = []
    if old_zr != 0 and new_zr != 0:
        diff_zr = new_zr - old_zr
        new_zr = new_zr + diff_zr
        sum_it = new_zr
        count_it = 1
    else:
        sum_it = 0
        count_it = 0
        for ii in range(mc):
            if ii < (mc - 1) and zs[source_ix+ii] != zs[source_ix+ii+1]:
                sum_it = sum_it + zrs[source_ix+ii]
                count_it = count_it + 1
                if ii == (mc - 2):
                    sum_it = sum_it + zrs[source_ix+ii+1]
                    count_it = count_it + 1
    temp_mean = sum_it / count_it
    for ii in range(mc):
        if ii < (mc - 1) and zs[source_ix+ii] == zs[source_ix+ii+1]:
            zrdup = find_furthest_value([zrs[source_ix+ii], zrs[source_ix+ii+1]], temp_mean)
            if zrdup == zrs[source_ix+ii]:
                duplicatez_source_ix.append(source_ix+ii)
                sum_it = sum_it + zrs[source_ix+ii+1]
            else:
                duplicatez_source_ix.append(source_ix+ii+1)
                sum_it = sum_it + zrs[source_ix+ii]
            count_it = count_it + 1
    temp_mean = sum_it / count_it
    return (temp_mean, duplicatez_source_ix)

def find_duplicate_z_using_zr(track, labels, df, debug=False):
    """Use z/r values to identify which hits in the track are invalid, when more than
    one hit has an identical 'z' co-ordinate."""
    duplicatez_ix = []
    hit_ix = np.where(labels == track)[0]

    if debug: print('hits: ' + str(hit_ix))
    # Need at least 4 values to be able to evaluate duplicate z-values.
    if len(hit_ix) < 4:
        if debug: print('Small hits! ' + str(len(hit_ix)))
        return duplicatez_ix

    df2 = df.loc[hit_ix]        
    df2 = df2.sort_values('z')
    hit_ix2 = df2.index.values # remember new indexes after sorting

    if debug: print(df2)
    xs = df2.x.values
    ys = df2.y.values
    zs = df2.z.values
    zrs = df2.zr.values
    volumes = df2.volume_id.values
    layers = df2.layer_id.values

    # The diff from one layer to the next within the same volume should
    # generally be relatively constant. Find the layer average for the
    # two layers before the layer that contains duplicate z-values.
    merged_layers = []
    merged_counts = []
    for layer in layers:
        if len(merged_layers) == 0 or layer != merged_layers[-1]:
            merged_layers.append(layer)
            merged_counts.append(1)
        else:
            merged_counts[-1] = merged_counts[-1] + 1

    if debug: print('Merged layers: ' + str(merged_layers))
    if debug: print('Merged counts: ' + str(merged_counts))
    old_zr = 0
    new_zr = 0
    source_ix = 0
    merged_zrs = []
    need_reverse_scan = False
    for mix, ml in enumerate(merged_layers):
        mc = merged_counts[mix]
        if all_zs_unique(zs, mc, source_ix):
            merged_zrs.append(np.mean(zrs[source_ix:source_ix+mc]))
        elif old_zr != 0 and new_zr != 0:
            (temp_mean, source_ix_dups) = find_dupzr_in_layer(old_zr, new_zr, zs, zrs, mc, source_ix)
            for src_ix in source_ix_dups:
                duplicatez_ix.append(hit_ix2[src_ix])
            merged_zrs.append(temp_mean)
        else:
            merged_zrs.append(0)
            need_reverse_scan = True

        old_zr = new_zr
        new_zr = merged_zrs[mix]
        source_ix = source_ix + mc

    if need_reverse_scan:
        old_zr = 0
        new_zr = 0
        source_ix = len(zs)
        for mix, ml in reversed(list(enumerate(merged_layers))):
            mc = merged_counts[mix]
            source_ix = source_ix - mc
            if merged_zrs[mix] == 0 and can_detect_dupzr(old_zr, new_zr, zs, mc, source_ix):
                (temp_mean, source_ix_dups) = find_dupzr_in_layer(old_zr, new_zr, zs, zrs, mc, source_ix)
                for src_ix in source_ix_dups:
                    duplicatez_ix.append(hit_ix2[src_ix])
                merged_zrs[mix] = temp_mean
            elif merged_zrs[mix] == 0:
                if debug: print('Rev. skipping, missing old_zr/new_zr, ix: ' + str(mix) + ', zs: ' + str(zs))

            old_zr = new_zr
            new_zr = merged_zrs[mix]
    

    if len(duplicatez_ix) > 0 and debug:
        print('Duplicatez found on track ' + str(track) + ', removed: ' + str(duplicatez_ix))

    return duplicatez_ix


def remove_track_outliers_slope(track, labels, hits, debug=False):
    """Compute slope between each pair of hits to identify suspected outliers."""
    # TODO pi, -pi discontinuity 
    final_outliers = []

    hhh_ix = np.where(labels == track)
    hhh_h = hits.loc[hhh_ix].sort_values('z')
    
    slopes_backward = []
    slopes_forward = []

    num_hits = len(hhh_h)
    # Only reliable with tracks >= 5 hits
    if num_hits < 5:
        return final_outliers

    if debug: print('backward:')
    for i in np.arange(num_hits-1,0,-1):
        a0 =  hhh_h.a0.values[i]
        a1 =  hhh_h.a0.values[i-1]
        r0 =  hhh_h.r.values[i]
        r1 =  hhh_h.r.values[i-1]
        if r0 == r1:
            r0 = r0 + 1e-8
        slope = (a0-a1)/(r0-r1) 
        slopes_backward.append(slope)
        if debug: print(hhh_h.hit_id.values[i], slope, a0)
        if i == 1:
            a0 = hhh_h.a0.values[0]
            a1 = hhh_h.a0.values[num_hits-1]
            r0 =  hhh_h.r.values[0]
            r1 =  hhh_h.r.values[num_hits-1]
            if r0 == r1:
                r0 = r0 + 1e-8
            slope = (a0-a1)/(r0-r1)
            slopes_backward.append(slope)
            if debug: print(hhh_h.hit_id.values[0], slope, a1)

    if debug: print('forward:')
    for i in np.arange(0,num_hits-1,1):
        a0 =  hhh_h.a0.values[i]
        a1 =  hhh_h.a0.values[i+1]
        r0 =  hhh_h.r.values[i]
        r1 =  hhh_h.r.values[i+1]
        if r0 == r1:
            r1 = r1 + 1e-8
        slope = (a1-a0)/(r1-r0) 
        slopes_forward.append(slope)
        if debug: print(hhh_h.hit_id.values[i], slope, a0)

        if i == num_hits-2:
            a0 = hhh_h.a0.values[0]
            a1 = hhh_h.a0.values[num_hits-1]
            r0 =  hhh_h.r.values[0]
            r1 =  hhh_h.r.values[num_hits-1]
            if r0 == r1:
                r1 = r1 + 1e-8
            slope = (a1-a0)/(r1-r0) 
            slopes_forward.append(slope)
            if debug: print(hhh_h.hit_id.values[num_hits-1], slope, a0)

    slopes_backward = np.asarray(slopes_backward)
    slopes_backward = np.reshape(slopes_backward, (-1, 1))
    slopes_forward = np.asarray(slopes_forward)
    slopes_forward = np.reshape(slopes_forward, (-1, 1))

    ss = StandardScaler()
    X_back = ss.fit_transform(slopes_backward)
    X_for = ss.fit_transform(slopes_forward)

    cl = DBSCAN(eps=0.0033, min_samples=1)
    outlier_labels_backward = cl.fit_predict(X_back)
    outlier_labels_forward = cl.fit_predict(X_for)

    if debug: print(outlier_labels_backward)
    if debug: print(outlier_labels_forward)

    track_counts = coll.Counter(outlier_labels_backward).most_common(1)
    most_common_id = track_counts[0][0]
    most_common_count = track_counts[0][1]

    outlier_indices_backward = []
    if most_common_count > 1 and len(np.unique(outlier_labels_forward)) < num_hits/2:
        for i in np.arange(num_hits-1,-1,-1):
            if outlier_labels_backward[i] != most_common_id:
                if debug: print(hhh_h.index.values[num_hits-1-i])
                outlier_indices_backward.append(hhh_h.index.values[num_hits-1-i])

    track_counts = coll.Counter(outlier_labels_forward).most_common(1)
    most_common_id = track_counts[0][0]
    most_common_count = track_counts[0][1]


    outlier_indices_forward = []
    if most_common_count > 1 and len(np.unique(outlier_labels_forward)) < num_hits/2:
        for i in np.arange(0,num_hits-1,1):
            if outlier_labels_forward[i] != most_common_id:
                if debug: print(hhh_h.index.values[i])
                outlier_indices_forward.append(hhh_h.index.values[i])


    outlier_candidates = list(set(outlier_indices_backward).intersection(outlier_indices_forward))


    if debug: print('before removal:' + str(outlier_candidates))

    for i in range(len(outlier_candidates)):
        candidate = hhh_h.loc[outlier_candidates[i]]
        found = False
        for index, row in hhh_h.iterrows():
            if np.absolute(candidate.z-row.z) == 0.5 and candidate.volume_id == row.volume_id \
            and candidate.layer_id == row.layer_id and candidate.module_id != row.module_id:
                # true hits
                if debug: print('true hit' + str(outlier_candidates[i]))
                found = True
        if found is False:
            final_outliers.append(outlier_candidates[i])

    if debug: print('new loutliers:' + str(final_outliers))

    # If we determine that half (or more) of the hits need to be removed, we may have messed
    # up, so do not return any outliers.
    max_removal_threshold = math.floor(num_hits/2)
    if len(final_outliers) >= max_removal_threshold:
        final_outliers = []

    return final_outliers

def pixel_matrix(pixel_cluster, show=False):
    """Calculate the cluster size and make a pixel matrix."""
    # cluster size
    min0 = min(pixel_cluster['ch0'])
    max0 = max(pixel_cluster['ch0'])
    min1 = min(pixel_cluster['ch1'])
    max1 = max(pixel_cluster['ch1'])
    # the matrix
    matrix = np.zeros(((max1-min1+3),(max0-min0+3)))
    for pixel in pixel_cluster.values :
        i0 = int(pixel[1]-min0+1)
        i1 = int(pixel[2]-min1+1)
        value = pixel[3]
        #deposit charge
        matrix[i1][i0] = value 
    # return the matris
    if show :
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.YlOrRd)
        plt.colorbar()
        plt.show()
    return matrix, max0-min0+1, max1-min1+1

def find_direction(matrix, axis=0,invalid_val=-1):
    #compare the left most non-zero value and the right most non-zero value
    mask = matrix!=0
    non_zero_idx = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    #print(non_zero_idx)
    i = invalid_val
    j = invalid_val
    direction = 0
    for index in range(len(non_zero_idx)):
        if non_zero_idx[index] != invalid_val:
            i = non_zero_idx[index]
        if non_zero_idx[-index-1] != invalid_val:
            j = non_zero_idx[-index-1]
        if i != invalid_val and j != invalid_val:
            direction = i - j
            break
            
    return direction
    
def find_celloutlier(track, labels, hits, cells):
    """Inspect cell information for each hit in the track to identify outliers.
    Outliers are those where the cell data suggests the particle is traveling
    in the opposite direction as other hits in the same track."""
    final_outliers = []

    hhh_ix = np.where(labels == track)
    hhh_h = hits.loc[hhh_ix].sort_values('z')
    
    def set_direction(hit_id):
        pixel_cluster = cells[cells['hit_id']==hit_id]
        matrix, width, length = pixel_matrix(pixel_cluster)
        direction = find_direction(matrix)
        if direction == 0 and matrix.shape[1]>3:
            direction = find_direction(matrix, axis=1)
        return direction
            
        
    hhh_h['direction'] = hhh_h.apply(lambda row: set_direction(row['hit_id']), axis=1)
    
    positive_hhh = hhh_h[hhh_h['direction'] > 0]
    negative_hhh = hhh_h[hhh_h['direction'] < 0]

    if len(negative_hhh) < len(positive_hhh) and len(negative_hhh) == 1:
        outlier_hhh = negative_hhh
    else:
        outlier_hhh = None
        
    if outlier_hhh is not None:
        final_outliers = outlier_hhh[outlier_hhh['volume_id'] > 9].hit_id.values - 1

    return final_outliers

    
def remove_track_outliers(track, labels, hits, cells, aggressive):
    """Remove all hits from tracks where the hit appears to be an outlier."""
    labels = np.copy(labels)
    found_bad_volume = 0
    found_bad_cell = 0
    found_bad_dimension = 0
    found_bad_slope = 0
    found_bad_z = 0
    found_bad_zr = 0

    if True:
        outlier_zr = zro.find_track_outliers_zr(track, labels, hits)
        if len(outlier_zr) > 0:
            #print('track ' + str(track) + ' zr outliers: ' + str(outlier_zr))
            found_bad_zr = found_bad_zr + len(outlier_zr)
            for oix in outlier_zr:
                labels[oix] = 0

    if aggressive:
        # Check if the sorted hits (on z-axis) go through the volumes
        # and layers in the expected order
        duplicatez_ix = find_duplicate_z_using_zr(track, labels, hits)
        if len(duplicatez_ix) > 0:
            #print('track ' + str(track) + ' duplicate z: ' + str(duplicatez_ix))
            found_bad_z = found_bad_z + len(duplicatez_ix)
            for bzix in duplicatez_ix:
                labels[bzix] = 0

    if aggressive:
        # Check the helix slope, discard hits that do not match
        outlier_slope_ix = remove_track_outliers_slope(track, labels, hits)
        if len(outlier_slope_ix) > 0:
            #print('track ' + str(track) + ' slope outliers: ' + str(outlier_slope_ix))
            found_bad_slope = found_bad_slope + len(outlier_slope_ix)
            for oix in outlier_slope_ix:
                labels[oix] = 0

    if False:#aggressive:# would be good to enable for aggressive outlier removal, but hurts score too much
        outlier_cell_ix = find_celloutlier(track, labels, hits, cells)
        if len(outlier_cell_ix) > 0:
            #print('track ' + str(track) + ' bad volume: ' + str(bad_volume_ix))
            found_bad_cell = found_bad_cell + len(outlier_cell_ix)
            for ocix in outlier_cell_ix:
                labels[ocix] = 0

    return (labels, found_bad_volume, found_bad_dimension, found_bad_z, found_bad_slope, found_bad_zr, found_bad_cell)

def remove_small_tracks(labels, smallest_track_size=2):
    """Remove small tracks that provide little value, can speed up other methods since they do not
    need to evualate these small tracks."""
    count_small_tracks = 0
    tracks, counts = np.unique(labels, return_counts=True)
    for track, count in zip(tracks, counts):
        if track != 0 and count < smallest_track_size:
            count_small_tracks = count_small_tracks + 1
            labels[labels == track] = 0
    return (labels, count_small_tracks)


def remove_outliers(labels, hits, cells, smallest_track_size=2, aggressive=False, print_counts=False):
    """Perform outlier identification and removal. Outlier removal will lower the LB score since there
    will be some false positives. Good outlier detection algorithms should have a false positive rate
    of 20% or less. Even though the LB score will be lower after outlier removal, the resulting tracks
    can typically be merged better, yielding a higher overall score after merging."""
    tracks = np.unique(labels)
    hits['z_abs'] = hits.z.abs()
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['a0'] = np.arctan2(hits.y,hits.x)
    hits['zr'] = hits['z'] / hits['r']
    count_rem_volume = 0
    count_rem_dimension = 0
    count_duplicatez = 0
    count_rem_slope = 0
    count_small_tracks = 0
    count_zr = 0
    count_cell = 0
    for track in tracks:
        if track == 0:
            continue
        track_hits = np.where(labels == track)[0]
        if len(track_hits) > 3:
            (labels, c1, c2, c3, c4, c5, c6) = remove_track_outliers(track, labels, hits, cells, aggressive)
            count_rem_volume = count_rem_volume + c1
            count_rem_dimension = count_rem_dimension + c2
            count_duplicatez = count_duplicatez + c3
            count_rem_slope = count_rem_slope + c4
            count_zr = count_zr + c5
            count_cell = count_cell + c6

    # Remove small tracks, we do not get any score for those. This is done
    # last, in case removing the outliers (above) removed enough hits
    # from a track to make them smaller than the threshold.
    (labels, count_small_tracks) = remove_small_tracks(labels, smallest_track_size=smallest_track_size)

    if print_counts:
        print('Total removed due to bad cells: ' + str(count_cell))
        print('Total removed due to bad volumes: ' + str(count_rem_volume))
        print('Total removed due to bad zr values: ' + str(count_zr))
        print('Total removed due to bad dimensions: ' + str(count_rem_dimension))
        print('Total removed due to duplicate zs: ' + str(count_duplicatez))
        print('Total removed due to bad slopes: ' + str(count_rem_slope))
        print('Total removed small tracks (<' + str(smallest_track_size) + ') hits: ' + str(count_small_tracks))

    return labels
