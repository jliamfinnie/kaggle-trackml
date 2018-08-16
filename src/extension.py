import numpy as np
import time as time
from sklearn.neighbors import KDTree
import collections as coll
import track_score as score
import straight_tracks as strt

def try_extend_single_hit_avoid_outliers(track, track_len, hit_ix, labels, hits):
    """
    This function tries to avoid extending our track when the target hit looks
    like it would likely be an outlier, for example if it contains the exact same z-value
    as the previous hit in the track, or if the new hit's zr value is significantly
    different than hits from the previous layer. If it looks like it could be a
    valid part of the given track, we compute scores (based on track length and number
    of estimated outliers) for the two tracks, and only assign the new hit to the
    track if it has a higher score than the track that the hit was previousls assigned to.
    """
    trk_ix = np.where(labels == track)[0]
    df = hits.loc[trk_ix]
    (z) = df[['z']].values.astype(np.float32).T
    hit_z = hits.loc[hit_ix].z
    hit_zr = hits.loc[hit_ix].zr
    hit_volume = hits.loc[hit_ix].volume_id
    hit_layer = hits.loc[hit_ix].layer_id.astype(np.int)
    #print('hit_layer: ' + str(hit_layer))
    if hit_z in z:
        #print('Already have z: ' + str(hit_z) + ', ' + str(z))
        return (labels, track_len)
    df = df.loc[(hits['volume_id'] == hit_volume)]
    df = df.sort_values('z')
    (x, y, z, zr) = df[['x', 'y', 'z', 'zr']].values.astype(np.float32).T
    layer = df.layer_id.values
    lmap = [0,0,0,0,1,0,2,0,3,0,4,0,5,0,6]
    #print(layer)
    (xs, ys, zrs, counts) = strt.generate_zr_layer_data(x, y, zr, layer, lmap)
    aix = lmap[hit_layer]
    if zrs[aix] != 0 and abs(zrs[aix]) > 1.0: # FIXME: Evaluate ones <= 1.0....
        #print('existing mean zr: ' + str(zrs[aix]) + ', hit_zr: ' + str(hit_zr))
        abs1 = abs(zrs[aix])
        abs2 = abs(hit_zr)
        if abs(zrs[aix]) < 1:
            min_abs = abs1 * 0.8
            max_abs = abs1 * 1.2
        elif abs(zrs[aix]) < 10:
            min_abs = abs1 * 0.9
            max_abs = abs1 * 1.1
        else:
            min_abs = abs1 * 0.99
            max_abs = abs1 * 1.01
        if abs2 < min_abs or abs2 > max_abs or (zrs[aix] > 0 and hit_zr < 0) or (zrs[aix] < 0 and hit_zr > 0):
            #print('SKIP: existing mean zr: ' + str(zrs[aix]) + ', hit_zr: ' + str(hit_zr) + ', min: ' + str(min_abs) + ', max: ' + str(max_abs))
            return (labels, track_len)

    outlier_modifier = 0.75
    orig_track = labels[hit_ix]
    labels[hit_ix] = track
    new_score = score.calculate_track_score(track, labels, hits, outlier_modifier=outlier_modifier, outlier_ix=hit_ix)
    labels[hit_ix] = orig_track
    if orig_track != 0:
        orig_score = score.calculate_track_score(orig_track, labels, hits, outlier_modifier=outlier_modifier, outlier_ix=hit_ix)
    else:
        orig_score = 0

    if new_score >= orig_score:
        labels[hit_ix] = track
        track_len = track_len + 1
                
    return (labels, track_len)

def try_extend_single_hit(track, track_len, hit_ix, labels, hits):
    """Assign the input hit to the input track if the input track is longer than the track
    that the hit was previously assigned to."""
    orig_track = labels[hit_ix]
    if orig_track == 0:
        labels[hit_ix] = track
    else:
        # If the hit is already occupied by another track, only take ownership
        # of the hit if our track is longer than the current-occupying track.
        orig_track_len = len(np.where(labels==orig_track)[0])
        if track_len > orig_track_len:
            labels[hit_ix] = track
            track_len = track_len + 1
                
    return (labels, track_len)


def _one_cone_slice(df, df1, angle, delta_angle, limit=0.04, num_neighbours=18, use_scoring=False):
    """Perform track extensions for a single cone slice."""
    min_num_neighbours = len(df1)
    if min_num_neighbours < 3: 
        return df

    hit_ids = df1.hit_id.values
    a,c,s,r,zr,z = df1[['a', 'c', 's', 'r_norm', 'zr', 'z']].values.T
    tree = KDTree(np.column_stack([c,s,r,zr]), metric='euclidean')

    track_ids = list(df1.track_id.unique())
    num_track_ids = len(track_ids)
    min_length=2

    labels = df.track_id.values
    label_track_counts = coll.Counter(df1.track_id.values).most_common(num_track_ids)
    
    for track_count in label_track_counts:
        p = track_count[0]
        if p == 0: continue

        idx = np.where(df1.track_id==p)[0]
        cur_track_len = len(idx)
        if cur_track_len<min_length: continue

        # Un-comment following code to find the truth particle ID for the track.
        #truth_ix = []
        #for ii in idx:
        #    truth_ix.append(hit_ids[ii] - 1)
        #tdf = truth.loc[truth_ix]
        #truth_count = coll.Counter(tdf.particle_id.values).most_common(1)
        #truth_particle_id = truth_count[0][0]
        #print('track: ' + str(p) + ', len: ' + str(len(idx)) + ', idx: ' + str(idx))
        #print('truth particle: ' + str(truth_particle_id) + ', count:' + str(truth_count[0][1]))

        if angle>0:
            idx = idx[np.argsort( z[idx])]
        else:
            idx = idx[np.argsort(-z[idx])]

## start and end points  ##
        idx0,idx1 = idx[0],idx[-1]
        a0 = a[idx0]
        a1 = a[idx1]
        r0 = r[idx0]
        r1 = r[idx1]
        c0 = c[idx0]
        c1 = c[idx1]
        s0 = s[idx0]
        s1 = s[idx1]
        zr0 = zr[idx0]
        zr1 = zr[idx1]

        da0 = a[idx[1]] - a[idx[0]]  #direction
        dr0 = r[idx[1]] - r[idx[0]]
        direction0 = np.arctan2(dr0,da0)

        da1 = a[idx[-1]] - a[idx[-2]]
        dr1 = r[idx[-1]] - r[idx[-2]]
        direction1 = np.arctan2(dr1,da1)

        ## extend start point
        ns = tree.query([[c0, s0, r0, zr0]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
        ns = np.concatenate(ns)

        direction = np.arctan2(r0 - r[ns], a0 - a[ns])
        diff = 1 - np.cos(direction - direction0)
        ns = ns[(r0 - r[ns] > 0.01) & (diff < (1 - np.cos(limit)))]
        for n in ns:
            df_ix = hit_ids[n] - 1
            # Un-comment this to see if we are extending the track properly
            #is_good = (truth.loc[df_ix, 'particle_id'] == truth_particle_id)
            ### TMP ###
            #if cur_track_len > 20: break
            ### TMP ###

            if use_scoring:
                (labels, cur_track_len) = try_extend_single_hit_avoid_outliers(p, cur_track_len, df_ix, labels, df)
            else:
                (labels, cur_track_len) = try_extend_single_hit(p, cur_track_len, df_ix, labels, df)

        ## extend end point
        ns = tree.query([[c1, s1, r1, zr1]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
        ns = np.concatenate(ns)

        direction = np.arctan2(r[ns] - r1, a[ns] - a1)
        diff = 1 - np.cos(direction - direction1)
  
        ns = ns[(r[ns] - r1 > 0.01) & (diff < (1 - np.cos(limit)))]
        for n in ns:  
            df_ix = hit_ids[n] - 1
            # Un-comment this to see if we are extending the track properly
            #is_good = (truth.loc[df_ix, 'particle_id'] == truth_particle_id)

            ### TMP ###
            #if cur_track_len > 23: break
            ### TMP ###

            if use_scoring:
                (labels, cur_track_len) = try_extend_single_hit_avoid_outliers(p, cur_track_len, df_ix, labels, df)
            else:
                (labels, cur_track_len) = try_extend_single_hit(p, cur_track_len, df_ix, labels, df)

    df['track_id'] = labels

    return df

def do_all_track_extensions(labels, hits, track_extension_limits, num_neighbours=18, use_scoring=False):
    """Perform track extensions using each of the specified proximity limits.
    When used with track_extension_limits=[0.02, 0.04, 0.06, 0.08, 0.10], this typcially
    provides a 0.05 to 0.15 improvement to the LB score."""
    time1 = time.time()
    df = hits.copy(deep=True)
    df['track_id'] = labels.tolist()
    df['r'] = np.sqrt(df.x**2 + df.y**2)
    df['r_norm'] = df.r / 1000
    df['zr'] = df.z / df.r
    df['arctan2'] = np.arctan2(df.z, df.r)
    for ix, limit in enumerate(track_extension_limits):
        df = extend(ix, df, do_swap=(ix%2==1), limit=(limit), num_neighbours=num_neighbours, use_scoring=use_scoring)
    time2 = time.time()
    print('Track extension took {:.3f} ms'.format((time2-time1)*1000.0))
    return df.track_id.values

def extend(iter, df, do_swap=False, limit=0.04, num_neighbours=18, use_scoring=False):
    """Perform track extensions for the given proximity limit."""
    if do_swap:
        df = df.assign(x = -df.x)
        df = df.assign(y = -df.y)

    df['a'] = np.arctan2(df.y, df.x)
    df['c'] = np.cos(df.a)
    df['s'] = np.sin(df.a)

    for angle in range(-90,90,1):

        print ('\r%d %f '%(iter,angle), end='',flush=True)
        df1 = df.loc[(df.arctan2>(angle-1.0)/180*np.pi) & (df.arctan2<(angle+1.0)/180*np.pi)]

        num_hits = len(df1)
        # Dynamically adjust the delta based on how many hits are found
        if num_hits > 2000:
            df1 = df.loc[(df.arctan2>(angle - 0.6 - 0.4)/180*np.pi) & (df.arctan2<(angle -0.6 + 0.4)/180*np.pi)]
            df = _one_cone_slice(df, df1, angle-0.6, 0.4, limit, num_neighbours, use_scoring)
            df1 = df.loc[(df.arctan2>(angle - 0.2 - 0.4)/180*np.pi) & (df.arctan2<(angle -0.2 + 0.4)/180*np.pi)]
            df = _one_cone_slice(df, df1, angle-0.2, 0.4, limit, num_neighbours, use_scoring)
            df1 = df.loc[(df.arctan2>(angle + 0.2 - 0.4)/180*np.pi) & (df.arctan2<(angle +0.2 + 0.4)/180*np.pi)]
            df = _one_cone_slice(df, df1, angle+0.2, 0.4, limit, num_neighbours, use_scoring)
            df1 = df.loc[(df.arctan2>(angle + 0.6 - 0.4)/180*np.pi) & (df.arctan2<(angle +0.6 + 0.4)/180*np.pi)]
            df = _one_cone_slice(df, df1, angle+0.6, 0.4, limit, num_neighbours, use_scoring)
        else:
            df = _one_cone_slice(df, df1, angle, 1, limit, num_neighbours, use_scoring)
           
    return df

       
