import numpy as np
import pandas as pd
import math
import collections as coll

def extend_straight_tracks(labels, hits):
    labels = np.copy(labels)
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['zr'] = hits['z'] / hits['r']

    # For the first round, try to lengthen any tracks based on expected
    # zr values in adjacent layers.
    tracks = np.unique(labels)
    for track in tracks:
        if track == 0: continue
        is_straight = is_straight_track(track, labels, hits)
        # Tracks that appear to be straight are more aggressive when finding
        # a hit in the next layer.
        labels = straight_track_extension(track, labels, hits, is_straight)

    # For the second round, only do the tracks that appear straight. These
    # are lengthened a second time in case non-straight tracks accidentally
    # took some hits away from the straight tracks.
    tracks = np.unique(labels)
    for track in tracks:
        if track == 0: continue
        if is_straight_track(track, labels, hits):
            labels = straight_track_extension(track, labels, hits, True)

    return labels

def get_expected_range(ix, xs, ys, zrs, use_largest_zrdiff=False):
    def get_min_max(old_val, diff):
        new_val = old_val - diff
        if new_val > old_val:
            min_val = old_val
            max_val = new_val - diff
        else:
            min_val = new_val - diff
            max_val = old_val
        return (min_val, new_val, max_val)

    # FIXME: Simple for now, expect linear changes
    # Should be enhanced to look at trends as well.
    xdiffs = np.diff(xs)
    ydiffs = np.diff(ys)
    zrdiffs = np.diff(zrs)
    # The difference in zr values is typically much smaller than the
    # difference in x and y values. Use the largest found zr diff,
    # instead of the adjacent zr diff (x and y use adjacent values).
    best_zrdiff = 0
    for iix, zr in enumerate(zrs):
        if iix < (len(zrs)-1) and zr != 0 and zrs[iix+1] != 0:
            if (zrdiffs[iix] < 0 and zrdiffs[iix] < best_zrdiff) or (zrdiffs[iix] > 0 and zrdiffs[iix] > best_zrdiff):
                best_zrdiff = zrdiffs[iix]

    if ix < (len(xs) - 1) and ix > 0 and xs[ix-1] != 0 and xs[ix+1] != 0:
        # Look at next 2 hits in track to find the expected value.
        x_min = min(xs[ix-1], xs[ix+1])
        x_max = max(xs[ix-1], xs[ix+1])
        x_exp = (x_min + x_max) / 2
        y_min = min(ys[ix-1], ys[ix+1])
        y_max = max(ys[ix-1], ys[ix+1])
        y_exp = (y_min + y_max) / 2
        zr_min = min(zrs[ix-1], zrs[ix+1])
        zr_max = max(zrs[ix-1], zrs[ix+1])
        zr_exp = (zr_min + zr_max) / 2
    elif ix < (len(xs) - 2) and xs[ix+1] != 0 and xs[ix+2] != 0:
        # Look at next 2 hits in track to find the expected value.
        (x_min, x_exp, x_max) = get_min_max(xs[ix+1], xdiffs[ix+1])
        (y_min, y_exp, y_max) = get_min_max(ys[ix+1], ydiffs[ix+1])
        if use_largest_zrdiff:
            # Use best_zrdiff to determine the min/max range, but actual zrdiff
            # to select most likely value
            (zr_min, zr_exp, zr_max) = get_min_max(zrs[ix+1], best_zrdiff)
            (_, zr_exp, _) = get_min_max(zrs[ix+1], zrdiffs[ix+1])
        else:
            (zr_min, zr_exp, zr_max) = get_min_max(zrs[ix+1], zrdiffs[ix+1])
    else:
        # Look at previous 2 hits in track to find the expected value.
        (x_min, x_exp, x_max) = get_min_max(xs[ix-1], -xdiffs[ix-2])
        (y_min, y_exp, y_max) = get_min_max(ys[ix-1], -ydiffs[ix-2])
        if use_largest_zrdiff:
            (zr_min, zr_exp, zr_max) = get_min_max(zrs[ix-1], -best_zrdiff)
            (_, zr_exp, _) = get_min_max(zrs[ix-1], -zrdiffs[ix-2])
        else:
            (zr_min, zr_exp, zr_max) = get_min_max(zrs[ix-1], -zrdiffs[ix-2])
    return (x_min, x_exp, x_max, y_min, y_exp, y_max, zr_min, zr_exp, zr_max)


def get_volume_switch_expected_zr_range(zr, factor=0.01):
    new_val_1 = zr * (1.00-factor)
    new_val_2 = zr * (1.00+factor)
    if new_val_1 > zr:
        min_val = new_val_2
        max_val = new_val_1
    else:
        min_val = new_val_1
        max_val = new_val_2
    return (min_val, max_val)

def is_weak_track(track, labels, volume, hits):
    """ Consider tracks with few hits (and no consecutive-layer hits) in our target volume
    as weak tracks we can steal hits from.
    """
    is_weak = False
    tix = np.where(labels == track)[0]
    df = hits.loc[tix]
    df.sort_values('z')
    volumes = df.volume_id.values
    layers = df.layer_id.values
    volume_layers = []
    for ix, vol in enumerate(volumes):
        if vol == volume:
            volume_layers.append(layers[ix])
    if len(volume_layers) <= 4:
        uniq_layers = np.unique(volume_layers)
        if len(uniq_layers) == 1 or (len(uniq_layers) == 2 and (uniq_layers[1] - uniq_layers[0] > 2)):
            is_weak = True

    return is_weak

def can_merge_tracks(track1, track2, labels, hits):
    # FIXME: Can be much smarter, for now, only consider merging both
    # tracks if they are both in the same volume, and do not have
    # any overlapping layers. Caller should verify that the tracks
    # are likely related, i.e. possibly by comparing zr values in
    # neighbouring layers to see if they are similar.
    merge_valid = False
    print_debug = False
    t1_ix = np.where(labels == track1)[0]
    t2_ix = np.where(labels == track2)[0]
    df1 = hits.loc[t1_ix]
    df2 = hits.loc[t2_ix]
    volume1 = np.unique(df1.volume_id.values)
    volume2 = np.unique(df2.volume_id.values)
    if len(volume1) == 1 and len(volume2) > 1 and volume1[0] in volume2:
        df2 = df2.loc[(df2['volume_id'] == volume1[0])]
        volume2 = np.unique(df2.volume_id.values)
    elif len(volume2) == 1 and len(volume1) > 1 and volume2[0] in volume1:
        df1 = df1.loc[(df1['volume_id'] == volume2[0])]
        volume1 = np.unique(df1.volume_id.values)
    if len(volume1) == 1 and len(volume2) == 1:
        if volume1[0] == volume2[0]:
            layers1 = np.unique(df1.layer_id.values)
            layers2 = np.unique(df2.layer_id.values)
            if (layers1[-1] + 2 == layers2[0]) or (layers2[-1] + 2 == layers1[0]):
                if print_debug: print('Merge tracks [' + str(track1) + ', ' + str(track2) + '] allowed, matching volume ' + str(volume1) + ' and non-overlapping adjacent layers: ' + str(layers1) + ', ' + str(layers2))
                merge_valid = True
            elif (len(layers1) == 1 and not layers1[0] in layers2) or (len(layers2) == 1 and not layers2[0] in layers1):
                if print_debug: print('Merge tracks [' + str(track1) + ', ' + str(track2) + '] allowed, matching volume ' + str(volume1) + ' and single-layer merge: ' + str(layers1) + ', ' + str(layers2))
                merge_valid = True
    return (merge_valid, t1_ix, t2_ix)

def find_nearest_zrs(dup_ixes, ixes, zs, zrs, ideal_zr, max_zrs=3):
    test_zrs = []
    test_zs = []
    test_ixes = []
    for aix, ix in enumerate(ixes):
        if ix in dup_ixes:
            test_zrs.append(zrs[aix])
            test_zs.append(zs[aix])
            test_ixes.append(ix)
    array = np.asarray(test_zrs)
    nearest_zrs_ix = []
    while len(test_zrs) > 0:
        array = np.asarray(test_zrs)
        idx = (np.abs(array - ideal_zr)).argmin()
        rem_z_value = test_zs[idx]
        nearest_zrs_ix.append(test_ixes[idx])
        test_zrs.pop(idx)
        test_zs.pop(idx)
        test_ixes.pop(idx)
        indexes = [i for i,z in enumerate(test_zs) if z == rem_z_value]
        if len(nearest_zrs_ix) >= max_zrs:
            break
        #print(len(test_zrs))
        #print(len(test_zs))
        #print(len(test_ixes))
        #print(len(indexes))
        #print(indexes)
        for ii in sorted(indexes, reverse=True):
            test_zrs.pop(ii)
            test_zs.pop(ii)
            test_ixes.pop(ii)
    return nearest_zrs_ix[0:max_zrs]

def select_best_zr_matches(track, labels, ix, xs, ys, zrs, px1, hits, aggressive_zr_estimation):
    # FIXME: Can be much smarter...
    print_debug = False
    px1_ixes = px1.index.values
    z_values = px1.z.values
    zr_values = px1.zr.values
    steal_ixes = []
    # First, assign any hits that do not contain duplicates
    duplicate_zs = []
    next_ix_is_dup = False
    for iix, z in enumerate(z_values):
        if next_ix_is_dup:
            duplicate_zs.append(px1_ixes[iix])
            next_ix_is_dup = False
        elif iix == (len(z_values) - 1) or z != z_values[iix+1]:
            steal_ixes.append(px1_ixes[iix])
        else:
            duplicate_zs.append(px1_ixes[iix])
            next_ix_is_dup = True
    if len(duplicate_zs) > 0:
        old_tracks = labels[duplicate_zs]
        #print('track: ' + str(track) + ' steal_ixes: ' + str(steal_ixes) + ', dup-zs: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks))
        unique_tracks = np.unique(old_tracks)
        if len(unique_tracks) == 1:
            steal_z_from_other_track = True
            (x_min, x_exp, x_max, y_min, y_exp, y_max, zr_min, zr_exp, zr_max) = get_expected_range(ix, xs, ys, zrs, aggressive_zr_estimation)
            steal_ixes = find_nearest_zrs(duplicate_zs, px1_ixes, z_values, zr_values, zr_exp)
            if print_debug: print('Track: ' + str(track) + ', stole ixes: ' + str(steal_ixes) + ' from all dup zrs: ' + str(duplicate_zs) + ', from tracks: ' + str(old_tracks))
        elif len(unique_tracks) == 2:
            # There are 2 separate tracks that the z values are distributed to.
            # First, check if we can merge.
            can_merge1 = False
            can_merge2 = False
            if unique_tracks[0] != 0:
                (can_merge1, m1_t1_ix, m1_t2_ix) = can_merge_tracks(track, unique_tracks[0], labels, hits)
            (can_merge2, m2_t1_ix, m2_t2_ix) = can_merge_tracks(track, unique_tracks[1], labels, hits)
            if can_merge1 and not can_merge2:
                #print('Track: ' + str(track) + ' can be merged with track1: ' + str(unique_tracks[0]))
                steal_ixes = m1_t2_ix
            elif can_merge2 and not can_merge1:
                #print('Track: ' + str(track) + ' can be merged with track2: ' + str(unique_tracks[1]))
                steal_ixes = m2_t2_ix
            else:
                if unique_tracks[0] == 0:
                    # just take the un-assigned hits, less risk of taking the wrong ones.
                    if print_debug: print('Track: ' + str(track) + ' taking un-assigned hits, ixes: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks))
                    for dupz in duplicate_zs:
                        if labels[dupz] == 0:
                            steal_ixes.append(dupz)
                else:
                    # See if there are up to 2 weak/invalid tracks we can steal from.
                    track_is_weak = []
                    for trk in unique_tracks:
                        track_is_weak.append(is_weak_track(trk, labels, px1.volume_id.values[0], hits))
                    weak_ixes = np.where(np.asarray(track_is_weak) == True)[0]
                    if len(weak_ixes) == 1 or len(weak_ixes) == 2:
                        weak_tracks = unique_tracks[weak_ixes]
                        for dupz in duplicate_zs:
                            if labels[dupz] in weak_tracks:
                                steal_ixes.append(dupz)
                        if print_debug: print('Track: ' + str(track) + ' taking hits from weak tracks: ' + str(weak_tracks) + ', hits: ' + str(steal_ixes))
                    else:
                        if print_debug: print('TODO! Track: ' + str(track) + ' cannot take hits, more work needed, ixes: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks) + ', weak_tracks: ' + str(track_is_weak))
        elif unique_tracks[0] == 0:
            # just take the un-assigned hits, less risk of taking the wrong ones.
            if print_debug: print('Track: ' + str(track) + ' many unique tracks, only taking un-assigned hits, ixes: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks))
            test_steal_ixes = steal_ixes
            steal_ixes = []
            for tsix in test_steal_ixes:
                if labels[tsix] == 0:
                    steal_ixes.append(tsix)
            for dupz in duplicate_zs:
                if labels[dupz] == 0:
                    steal_ixes.append(dupz)
        else:
            # See if there are up to 2 weak/invalid tracks we can steal from.
            track_is_weak = []
            for trk in unique_tracks:
                track_is_weak.append(is_weak_track(trk, labels, px1.volume_id.values[0], hits))
            weak_ixes = np.where(np.asarray(track_is_weak) == True)[0]
            if len(weak_ixes) == 1 or len(weak_ixes) == 2:
                weak_tracks = unique_tracks[weak_ixes]
                for dupz in duplicate_zs:
                    if labels[dupz] in weak_tracks:
                        steal_ixes.append(dupz)
                if print_debug: print('Track: ' + str(track) + ' taking hits from weak tracks: ' + str(weak_tracks) + ', hits: ' + str(steal_ixes))
            else:
                if print_debug: print('TODO! Track: ' + str(track) + ', too many tracks to steal duplicate zs from, ignoring dups: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks) + ', weak_tracks: ' + str(track_is_weak))

    #print(px1)
    return steal_ixes

def generate_zr_layer_data(x, y, zr, layer, lmap):
    xs = [0,0,0,0,0,0,0]
    ys = [0,0,0,0,0,0,0]
    zrs = [0,0,0,0,0,0,0]
    counts = [0,0,0,0,0,0,0]
    for ix, l in enumerate(layer):
        aix = lmap[l]
        counts[aix] = counts[aix] + 1
        xs[aix] = xs[aix] + x[ix]
        ys[aix] = ys[aix] + y[ix]
        zrs[aix] = zrs[aix] + zr[ix]
    for ix, count in enumerate(counts):
        if count != 0:
            xs[ix] = xs[ix] / count
            ys[ix] = ys[ix] / count
            zrs[ix] = zrs[ix] / count
    return (xs, ys, zrs, counts)

def one_round_straight_track_extension(track, labels, hits, aggressive_zr_estimation):
    print_debug = False
    more_rounds_possible = False
    hit_ix = np.where(labels == track)[0]
    if len(hit_ix) == 0:
        return (more_rounds_possible, labels)
    df = hits.loc[hit_ix]
    df = df.loc[(df['volume_id'] == 7) | (df['volume_id'] == 7) | (df['volume_id'] == 9)]
    msg = 'Track: ' + str(track) + ', '
    if not np.all(df.volume_id.values == 7) and not np.all(df.volume_id.values == 9) and not np.all(df.volume_id.values == 13):
        # FIXME: Future improvement, handle other volumes, and handle
        # tracks that span volumes.
        #print(msg + 'Can only lengthen straight tracks in volume 9, found: ' + str(df.volume_id.values))
        return (more_rounds_possible, labels)
    df = df.sort_values('z')
    x,y,zr = df[['x', 'y', 'zr']].values.astype(np.float32).T
    volume,layer = df[['volume_id', 'layer_id' ]].values.T
    #  indexes:  [2->0,4->1,6->2,8->3,10->4,12->5,14->6]
    lmap = [0,0,0,0,1,0,2,0,3,0,4,0,5,0,6]
    all_layers = [2,4,6,8,10,12,14]
    uniq_layers = np.unique(layer)
    uniq_volumes = np.unique(volume)
    if len(uniq_volumes) > 1:
        # FIXME: Future improvement, handle tracks across volumes
        #print(msg + 'All hits must be in same volume, volumes found: ' + str(uniq_volumes))
        return (more_rounds_possible, labels)
    elif np.array_equal(all_layers, uniq_layers):
        # FIXME: Future improvement, we can have multiple hits per layer,
        # check if we are missing some hits. Hits within the same layer
        # should have very small deltas
        #print(msg + 'All layers already have at least one hit')
        return (more_rounds_possible, labels)
    elif len(uniq_layers) == 1:
        #print(msg + 'Only one layer defined, unable to determine trends for extension')
        return (more_rounds_possible, labels)

    (xs, ys, zrs, counts) = generate_zr_layer_data(x, y, zr, layer, lmap)
    #print(msg + 'xs: ' + str(xs))
    #print(msg + 'ys: ' + str(ys))
    #print(msg + 'zrs: ' + str(zrs))
    for ix, l in enumerate(all_layers):
        if xs[ix] == 0:
            #if (ix >= 2 and xs[ix-1] != 0 and xs[ix-2] != 0) or (ix < (len(xs) - 2) and xs[ix+1] != 0 and xs[ix+2] != 0):
            if ((ix >= 2 and xs[ix-1] != 0 and xs[ix-2] != 0) or (ix < (len(xs) - 2) and xs[ix+1] != 0 and xs[ix+2] != 0)) or (ix >= 1 and ix < (len(xs) - 1) and xs[ix-1] != 0 and xs[ix+1] != 0):
                (x_min, x_exp, x_max, y_min, y_exp, y_max, zr_min, zr_exp, zr_max) = get_expected_range(ix, xs, ys, zrs, aggressive_zr_estimation)
                # DO IT!
                #print('x: ' + str(x_min) + ', ' + str(x_max) + ', y: ' + str(y_min) + ', ' + str(y_max) + ', zr: ' + str(zr_min) + ', ' + str(zr_max))
                possible_matches = hits.loc[(hits['y'] > y_min) & (hits['y'] < y_max) & (hits['x'] > x_min) & (hits['x'] < x_max) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == uniq_volumes[0]) & (hits['layer_id'] == l)]
                possible_matches = possible_matches.sort_values('z')
                px1 = possible_matches[['x','y', 'z', 'zr', 'volume_id', 'layer_id', 'module_id']]
                msg2 = msg + 'zr_exp: ' + str(zr_exp) + ', '
                if len(px1) >= 2 and len(np.unique(px1.z.values)) < len(px1) and len(np.unique(px1.z.values)) <= 3:
                    steal_ixs = select_best_zr_matches(track, labels, ix, xs, ys, zrs, px1, hits, aggressive_zr_estimation)
                    if len(steal_ixs) > 0:
                        # Assign hits!
                        #print(steal_ixs)
                        #print(labels[steal_ixs])
                        if print_debug: print(msg2 + 'assigning dup-z hits: ' + str(steal_ixs))
                        labels[steal_ixs] = track
                        more_rounds_possible = True
                        #print(labels[steal_ixs])
                        #print(px1)
                elif len(px1) >= 1 and len(px1) <= 3:
                    #(can_merge, t1_ix, t2_ix) = can_merge_tracks(track, unique_tracks[0], labels, hits)
                    # Assign hits!
                    steal_ixs = px1.index.values
                    steal_from_tracks = np.unique(labels[steal_ixs])
                    can_merge = False
                    if len(steal_from_tracks) == 1:
                        (can_merge, t1_ix, t2_ix) = can_merge_tracks(track, steal_from_tracks[0], labels, hits)
                    if can_merge:
                        if print_debug: print(msg2 + 'assigning hits: ' + str(t2_ix))
                        labels[t2_ix] = track
                    else:
                        #print(steal_ixs)
                        #print(labels[steal_ixs])
                        if print_debug: print(msg2 + 'assigning hits: ' + str(steal_ixs) + ' from tracks: ' + str(labels[steal_ixs]))
                        labels[steal_ixs] = track
                        #print(labels[steal_ixs])
                    more_rounds_possible = True
                    #print(px1)
                elif len(px1) > 0:
                    match_ixs = px1.index.values
                    # See if there are up to 2 weak/invalid tracks we can steal from.
                    unique_tracks = np.unique(labels[match_ixs])
                    track_is_weak = []
                    for trk in unique_tracks:
                        track_is_weak.append(is_weak_track(trk, labels, px1.volume_id.values[0], hits))
                    weak_ixes = np.where(np.asarray(track_is_weak) == True)[0]
                    if len(weak_ixes) == 1 or len(weak_ixes) == 2:
                        weak_tracks = unique_tracks[weak_ixes]
                        steal_ixs = []
                        for mix in match_ixs:
                            if labels[mix] in weak_tracks:
                                steal_ixs.append(mix)
                        labels[steal_ixs] = track
                        more_rounds_possible = True
                        if print_debug: print(msg2 + 'assigning hits: ' + str(steal_ixs) + ' stolen from weak tracks: ' + str(weak_tracks))
                    else:
                        if print_debug: print(msg2 + str(len(px1)) + ' possible matches')
                        if print_debug: print(msg2 + str(match_ixs))
                        if print_debug: print(msg2 + str(labels[match_ixs]))

    return (more_rounds_possible, labels)

def cleanse_straight_track(track, labels, hits):
    hit_ix = np.where(labels == track)[0]
    if len(hit_ix) == 0:
        return labels
    df = hits.loc[hit_ix]
    msg = 'Track: ' + str(track) + ', '
    if not np.all(df.volume_id.values == 12) and not np.all(df.volume_id.values == 8):
        # FIXME: Future improvement, handle other volumes, and handle
        # tracks that span volumes.
        #print(msg + 'Can only lengthen straight tracks in volume 9, found: ' + str(df.volume_id.values))
        return (labels)
    df = df.sort_values('z')
    hit_ix2 = df.index.values
    x,y,zr = df[['x', 'y', 'zr']].values.astype(np.float32).T
    volume,layer = df[['volume_id', 'layer_id' ]].values.T
    #  indexes:  [2->0,4->1,6->2,8->3,10->4,12->5,14->6]
    lmap = [0,0,0,0,1,0,2,0,3,0,4,0,5,0,6]
    all_layers = [2,4,6,8,10,12,14]
    uniq_layers = np.unique(layer)
    uniq_volumes = np.unique(volume)
    xs = [0,0,0,0,0,0,0]
    ys = [0,0,0,0,0,0,0]
    zrs = [0,0,0,0,0,0,0]
    counts = [0,0,0,0,0,0,0]
    # FIXME: LIAM: Within a single layer, only allow 0.5% variation?
    for ix, l in enumerate(layer):
        aix = lmap[l]
        counts[aix] = counts[aix] + 1
        xs[aix] = xs[aix] + x[ix]
        ys[aix] = ys[aix] + y[ix]
        zrs[aix] = zrs[aix] + zr[ix]
    for ix, count in enumerate(counts):
        if count != 0:
            xs[ix] = xs[ix] / count
            ys[ix] = ys[ix] / count
            zrs[ix] = zrs[ix] / count

    # DOES NOT WORK! 
    #if len(uniq_layers) == 3:
    #    # Favour keeping beginning of track, cut out high end if it looks wrong
    #    # sample zrs: 6.16, 6.22, 6.05
    #    if zrs[1] > zrs[0] and zrs[2] < (0.995*zrs[1]):
    #        print(msg + ' is a positive possible cleansing target')
    #    elif zrs[1] < zrs[0] and zrs[2] > (0.995*zrs[1]):
    #        print(msg + ' is a negative possible cleansing target')

    return labels

def merge_with_other_volumes(track, labels, hits):
    hit_ix = np.where(labels == track)[0]
    if len(hit_ix) == 0:
        return labels
    df = hits.loc[hit_ix]
    msg = 'Track: ' + str(track) + ', '
    if not np.all(df.volume_id.values == 7) and not np.all(df.volume_id.values == 9):
        # FIXME: Future improvement, handle other volumes, and handle
        # tracks that span volumes.
        #print(msg + 'Can only lengthen straight tracks in volume 9, found: ' + str(df.volume_id.values))
        return (labels)
    df = df.sort_values('z')
    x,y,zr = df[['x', 'y', 'zr']].values.astype(np.float32).T
    volume,layer = df[['volume_id', 'layer_id' ]].values.T
    #  indexes:  [2->0,4->1,6->2,8->3,10->4,12->5,14->6]
    lmap = [0,0,0,0,1,0,2,0,3,0,4,0,5,0,6]
    all_layers = [2,4,6,8,10,12,14]
    uniq_layers = np.unique(layer)
    uniq_volumes = np.unique(volume)
    if len(uniq_volumes) > 1:
        # FIXME: Future improvement, handle tracks across volumes
        #print(msg + 'All hits must be in same volume, volumes found: ' + str(uniq_volumes))
        return (labels)
    elif len(uniq_layers) == 1:
        #print(msg + 'Only one layer defined, unable to determine trends for extension')
        return (labels)

    find_zr_ix = 0
    attempt_vol12_merge = False
    print_debug = False
    (xs, ys, zrs, counts) = generate_zr_layer_data(x, y, zr, layer, lmap)
    if uniq_volumes[0] == 7 and zrs[0] != 0 and zrs[1] != 0:
        #print(msg + 'searching for extension from 7.2 down to 12.4')
        xs.insert(0,0)
        ys.insert(0,0)
        zrs.insert(0,0)
        counts.insert(0,0)
        attempt_vol12_merge = True
        target_layer_id = 4
    elif uniq_volumes[0] == 7 and zrs[1] != 0 and zrs[2] != 0:
        #print(msg + 'searching for extension from 7.4 down to 12.6)
        attempt_vol12_merge = True
        target_layer_id = 6
        #print_debug = True

    if attempt_vol12_merge:
        (x_min, x_exp, x_max, y_min, y_exp, y_max, zr_min, zr_exp, zr_max) = get_expected_range(find_zr_ix, xs, ys, zrs, use_largest_zrdiff=False)
        #print('x: ' + str(x_min) + ', ' + str(x_max) + ', y: ' + str(y_min) + ', ' + str(y_max) + ', zr: ' + str(zr_min) + ', ' + str(zr_max))
        #possible_matches = hits.loc[(hits['y'] > y_min) & (hits['y'] < y_max) & (hits['x'] > x_min) & (hits['x'] < x_max) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == 12) & (hits['layer_id'] == 4)]
        (zr_min, zr_max) = get_volume_switch_expected_zr_range(zrs[1], factor=0.02)
        if xs[1] > xs[2] and ys[1] > ys[2]:
            possible_matches = hits.loc[(hits['x'] > xs[1]) & (hits['y'] > ys[1]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == 12) & (hits['layer_id'] == target_layer_id)]
        elif xs[1] > xs[2] and ys[1] < ys[2]:
            possible_matches = hits.loc[(hits['x'] > xs[1]) & (hits['y'] < ys[1]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == 12) & (hits['layer_id'] == target_layer_id)]
        elif xs[1] < xs[2] and ys[1] > ys[2]:
            possible_matches = hits.loc[(hits['x'] < xs[1]) & (hits['y'] > ys[1]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == 12) & (hits['layer_id'] == target_layer_id)]
        else:
            possible_matches = hits.loc[(hits['x'] < xs[1]) & (hits['y'] < ys[1]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == 12) & (hits['layer_id'] == target_layer_id)]

        possible_matches = possible_matches.sort_values('z')
        px1 = possible_matches[['x','y', 'z', 'zr', 'volume_id', 'layer_id', 'module_id']]
        msg2 = msg + 'matches: ' + str(len(px1)) + ', zr_min: ' + str(zr_min) + ', zr_exp: ' + str(zr_exp) + ', zr_max: ' + str(zr_max) + ', '
        if print_debug: print(msg2 + 'x_min: ' + str(x_min) + ', y_min: ' + str(y_min))
        #print(msg + 'searching for extension from 7.2 down to 12.4, targets: ' + str(unique_tracks))
        if len(px1) > 0 and len(px1) <= 15:
            old_tracks = labels[px1.index.values]
            #print('track: ' + str(track) + ' steal_ixes: ' + str(steal_ixes) + ', dup-zs: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks))
            unique_tracks = np.unique(old_tracks)
            if print_debug: print(msg2 + 'searching for extension from 7.2 down to 12.4, targets: ' + str(unique_tracks))
            steal_ixes_track0 = []
            steal_ixes_merge = []
            for utrk in unique_tracks:
                if utrk == 0:
                    utrk_hits_ix = []
                    for iix in px1.index.values:
                        if labels[iix] == 0:
                            utrk_hits_ix.append(iix)
                else:
                    utrk_hits_ix = np.where(labels == utrk)[0]
                utrk_hits = hits.loc[utrk_hits_ix]
                if utrk == 0:
                    px2 = utrk_hits[['x','y', 'z', 'zr', 'volume_id', 'layer_id']]
                    if print_debug: print(px2)
                    px2 = px2.sort_values('z')
                    dup_ixes = px2.index.values
                    ixes = dup_ixes
                    target_zs = px2.z.values
                    target_zrs = px2.zr.values
                    ideal_zr = zr_exp
                    steal_ixes_track0 = find_nearest_zrs(dup_ixes, ixes, target_zs, target_zrs, ideal_zr, max_zrs=1)
                    if print_debug: print(msg2 + 'can steal hit: ' + str(steal_ixes_track0))
                else:
                    all_target_volumes = utrk_hits.volume_id.values
                    uniq_target_volumes = np.unique(all_target_volumes)
                    all_target_layers = utrk_hits.layer_id.values
                    all_target_indexes = utrk_hits.index.values
                    if len(uniq_target_volumes) == 1 and np.unique(all_target_layers)[-1] <= 6:
                        if print_debug: print(msg2 + 'can steal track ' + str(utrk))
                        #print(utrk_hits)
                        steal_ixes_merge = all_target_indexes
                        break
                    else:
                        if print_debug: print(msg2 + 'no perfect match for track ' + str(utrk) + ', running more checks....')
                        count_vol7 = 0
                        max_vol7_layer = 0
                        for iix, v in enumerate(all_target_volumes):
                            if v == 12:
                                steal_ixes_merge.append(all_target_indexes[iix])
                            elif v == 7:
                                count_vol7 = count_vol7 + 1
                                max_vol7_layer = max(max_vol7_layer, all_target_layers[iix])
                        if count_vol7 <= 6 and max_vol7_layer <= 4:
                            if print_debug: print(msg2 + 'short vol7 track ' + str(utrk) + ', can steal vol 12 hits.')
                            break
                        else:
                            if print_debug: print(msg2 + 'too many vol7 hits in track ' + str(utrk) + ', ' + str(count_vol7) + ', ' + str(max_vol7_layer))
                            steal_ixes_merge = []
                            
            # FIXME: Can potentially make this smarter if we find multiple
            # tracks we can merge with, to only merge with the one closest
            # to our ideal zr value. For now, just merge with first track
            # we find that seems like a candidate
            if len(steal_ixes_merge) > 0:
                labels[steal_ixes_merge] = track
            elif len(steal_ixes_track0) > 0:
                labels[steal_ixes_track0] = track

    find_zr_ix = len(zrs)
    attempt_volnext_merge = False
    print_debug = False
    factor = 0.04
    if ((uniq_volumes[0] == 7) or (uniq_volumes[0] == 9)) and zrs[-1] != 0 and zrs[-2] != 0:
        #print(msg + 'searching for extension from 7.14 up to 8.2')
        xs.append(0)
        ys.append(0)
        zrs.append(0)
        counts.append(0)
        attempt_volnext_merge = True
        #print_debug = True
        if uniq_volumes[0] == 7:
            target_volume_id = 8
            target_layer_id = 2
        else:
            target_volume_id = 14
            target_layer_id = 12

    if attempt_volnext_merge:
        (x_min, x_exp, x_max, y_min, y_exp, y_max, zr_min, zr_exp, zr_max) = get_expected_range(find_zr_ix, xs, ys, zrs, use_largest_zrdiff=False)
        #print('x: ' + str(x_min) + ', ' + str(x_max) + ', y: ' + str(y_min) + ', ' + str(y_max) + ', zr: ' + str(zr_min) + ', ' + str(zr_max))
        #possible_matches = hits.loc[(hits['y'] > y_min) & (hits['y'] < y_max) & (hits['x'] > x_min) & (hits['x'] < x_max) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == 12) & (hits['layer_id'] == 4)]
        (zr_min, zr_max) = get_volume_switch_expected_zr_range(zrs[-2], factor=factor)
        if xs[-2] > xs[-3] and ys[-2] > ys[-3]:
            possible_matches = hits.loc[(hits['x'] > xs[-2]) & (hits['y'] > ys[-2]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == target_volume_id) & (hits['layer_id'] == target_layer_id)]
        elif xs[-2] > xs[-3] and ys[-2] < ys[-3]:
            possible_matches = hits.loc[(hits['x'] > xs[-2]) & (hits['y'] < ys[-2]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == target_volume_id) & (hits['layer_id'] == target_layer_id)]
        elif xs[-2] < xs[-3] and ys[-2] > ys[-3]:
            possible_matches = hits.loc[(hits['x'] < xs[-2]) & (hits['y'] > ys[-2]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == target_volume_id) & (hits['layer_id'] == target_layer_id)]
        else:
            possible_matches = hits.loc[(hits['x'] < xs[-2]) & (hits['y'] < ys[-2]) & (hits['zr'] > zr_min) & (hits['zr'] < zr_max) & (hits['volume_id'] == target_volume_id) & (hits['layer_id'] == target_layer_id)]

        possible_matches = possible_matches.sort_values('z')
        px1 = possible_matches[['x','y', 'z', 'zr', 'volume_id', 'layer_id', 'module_id']]
        msg2 = msg + 'matches: ' + str(len(px1)) + ', zr_min: ' + str(zr_min) + ', zr_exp: ' + str(zr_exp) + ', zr_max: ' + str(zr_max) + ', '
        if print_debug: print(msg2 + 'x_min: ' + str(x_min) + ', y_min: ' + str(y_min))
        #print(msg + 'searching for extension from 7.2 down to 12.4, targets: ' + str(unique_tracks))
        if len(px1) > 0 and len(px1) <= 15:
            old_tracks = labels[px1.index.values]
            #print('track: ' + str(track) + ' steal_ixes: ' + str(steal_ixes) + ', dup-zs: ' + str(duplicate_zs) + ', old tracks: ' + str(old_tracks))
            unique_tracks = np.unique(old_tracks)
            if print_debug: print(msg2 + 'searching for extension from ' + str(uniq_volumes[0]) + '.14 up to ' + str(target_volume_id) + '.' + str(target_layer_id) + ', target tracks: ' + str(unique_tracks) + ', target hits: ' + str(px1.index.values))
            steal_ixes_track0 = []
            steal_ixes_merge = []
            for utrk in unique_tracks:
                if utrk == 0:
                    utrk_hits_ix = []
                    for iix in px1.index.values:
                        if labels[iix] == 0:
                            utrk_hits_ix.append(iix)
                else:
                    utrk_hits_ix = np.where(labels == utrk)[0]
                utrk_hits = hits.loc[utrk_hits_ix]
                if utrk == 0:
                    px2 = utrk_hits[['x','y', 'z', 'zr', 'volume_id', 'layer_id']]
                    if print_debug: print(px2)
                    px2 = px2.sort_values('z')
                    dup_ixes = px2.index.values
                    ixes = dup_ixes
                    target_zs = px2.z.values
                    target_zrs = px2.zr.values
                    ideal_zr = zr_exp
                    steal_ixes_track0 = find_nearest_zrs(dup_ixes, ixes, target_zs, target_zrs, ideal_zr, max_zrs=2)
                    if print_debug: print(msg2 + 'can steal hit: ' + str(steal_ixes_track0))
                else:
                    all_target_volumes = utrk_hits.volume_id.values
                    uniq_target_volumes = np.unique(all_target_volumes)
                    all_target_layers = utrk_hits.layer_id.values
                    all_target_indexes = utrk_hits.index.values
                    if len(uniq_target_volumes) == 1 and len(np.unique(all_target_layers)) <= 2:
                        if print_debug: print(msg2 + 'can steal track ' + str(utrk))
                        #print(utrk_hits)
                        steal_ixes_merge = all_target_indexes
                        break
                    else:
                        if print_debug: print(msg2 + 'no perfect match for track ' + str(utrk) + ', running more checks....')
                        count_my_volume = 0
                        max_my_volume_layer = 0
                        for iix, v in enumerate(all_target_volumes):
                            if v == target_volume_id:
                                steal_ixes_merge.append(all_target_indexes[iix])
                            elif v == uniq_volumes[0]:
                                count_my_volume = count_my_volume + 1
                                max_my_volume_layer = max(max_my_volume_layer, all_target_layers[iix])
                        if count_my_volume <= 4: # ???  and max_vol7_layer <= 4:
                            if print_debug: print(msg2 + 'short vol7 track ' + str(utrk) + ', can steal vol 12 hits.')
                            break
                        else:
                            if print_debug: print(msg2 + 'too many vol7 hits in track ' + str(utrk) + ', ' + str(count_my_volume) + ', ' + str(max_my_volume_layer))
                            steal_ixes_merge = []
                            
            # FIXME: Can potentially make this smarter if we find multiple
            # tracks we can merge with, to only merge with the one closest
            # to our ideal zr value. For now, just merge with first track
            # we find that seems like a candidate
            if len(steal_ixes_merge) > 0:
                labels[steal_ixes_merge] = track
            elif len(steal_ixes_track0) > 0:
                labels[steal_ixes_track0] = track

    return labels

def straight_track_extension(track, labels, hits, aggressive_zr_estimation):
    #labels = np.copy(labels)
    # No good cleanse operation yet....
    # Would be good to find cases where two hits in the same layer are
    # far apart, tell which one is likely outlier, and remove it.
    #labels = cleanse_straight_track(track, labels, hits)
    more_rounds = True
    while more_rounds:
        (more_rounds, labels) = one_round_straight_track_extension(track, labels, hits, aggressive_zr_estimation)
    labels = merge_with_other_volumes(track, labels, hits)
    return labels

def compare_track_to_truth(track, labels, hits, truth):
    hit_ix = np.where(labels == track)[0]
    df = hits.loc[hit_ix]
    df = df.sort_values('z')
    dfx1 = df[['x','y', 'z', 'zr', 'volume_id', 'layer_id']]

    tdf = truth.loc[hit_ix]
    truth_count = coll.Counter(tdf.particle_id.values).most_common(2)
    truth_particle_id = truth_count[0][0]
    if truth_particle_id == 0 and len(truth_count) > 1:
        truth_particle_id = truth_count[1][0]
    tdf2 = truth.loc[(truth.particle_id == truth_particle_id)]
    tdf2 = tdf2.sort_values('tz')

    arr_s1 = np.copy(hit_ix)
    arr_s1.sort()
    arr_s2 = np.copy(tdf2.index.values)
    arr_s2.sort()
    if np.array_equal(arr_s1, arr_s2):
        print('Equal!')
        print(arr_s1)
        print(arr_s2)
        print(dfx1)
    else:
        print('Detected track: ' + str(arr_s1))
        print('Truth track:    ' + str(arr_s2))
        print(dfx1)

        df3 = hits.loc[tdf2.index.values]
        dfx3 = df3[['x','y', 'z', 'zr', 'volume_id', 'layer_id']]
        print(dfx3)

def check_if_zr_straight(zr_values):
    if len(zr_values) < 3:
        is_straight = 0
    else:
        is_straight = 1
        mean_zr = zr_values.mean()
        num_outliers = 0
        if mean_zr < 0:
            allowed_min = mean_zr * 1.02
            allowed_max = mean_zr * 0.98
        else:
            allowed_min = mean_zr * 0.98
            allowed_max = mean_zr * 1.02
        # FIXME: Ignores outliers for now, tracks with outliers will
        # not likely be considered 'straight'.
        for zr_value in zr_values:
            if zr_value < allowed_min or zr_value > allowed_max:
                num_outliers = num_outliers + 1
                is_straight = 0
    return (is_straight, num_outliers)

def is_straight_track(track, labels, hits):
    is_straight = 0
    hit_ix = np.where(labels == track)[0]

    if len(hit_ix) > 2:
        df = hits.loc[hit_ix]
        df = df.sort_values('z')

        zr = df.zr.values
        (is_straight, num_outliers) = check_if_zr_straight(zr)
        # Allow a small number of outliers
        if not is_straight and num_outliers <= 2 and len(hit_ix) > 4:
            is_straight = 1

    return is_straight
