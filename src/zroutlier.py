import numpy as np
import pandas as pd
import math
import collections as coll

def classify_zr_shape(abs_zrs, diff_zrs):
    """
    5 shapes:
     0: unknown (no trend identified)
     1: increasing values
     2: decreasing values
     3: hill (increase then decrease)
     4: valley (decrease then increase)
    """
    def slice_diffs(diff_zrs):
        slices = []
        len_diffs = len(diff_zrs)
        if len_diffs > 12:
            len_part_diffs = int(len_diffs/3)
            slices.append(diff_zrs[0:len_part_diffs])
            slices.append(diff_zrs[len_part_diffs:2*len_part_diffs])
            slices.append(diff_zrs[2*len_part_diffs:])
        else:
            len_part_diffs = int(len_diffs/2)
            slices.append(diff_zrs[0:len_part_diffs])
            slices.append(diff_zrs[len_part_diffs:])
        return slices

    def trend_increase(diff_zrs):
        ret = False
        len_diffs = len(diff_zrs)
        if np.all(diff_zrs >= 0):
            ret = True
        elif len(np.where(diff_zrs >= 0)[0]) >= math.ceil(0.6*len_diffs):
            slices = slice_diffs(diff_zrs)
            ret = True
            for sample in slices:
                if len(np.where(sample >= 0)[0]) < math.ceil(0.5*len(sample)):
                    ret = False
                    break
        return ret

    def trend_decrease(diff_zrs):
        ret = False
        len_diffs = len(diff_zrs)
        if np.all(diff_zrs <= 0):
            ret = True
        elif len(np.where(diff_zrs <= 0)[0]) >= math.ceil(0.6*len_diffs):
            slices = slice_diffs(diff_zrs)
            ret = True
            for sample in slices:
                if len(np.where(sample <= 0)[0]) < math.ceil(0.5*len(sample)):
                    ret = False
                    break
        return ret

    def trend_hill(diff_zrs):
        ret = False
        slices = slice_diffs(diff_zrs)
        if trend_increase(slices[0]):
            for ix, sample in enumerate(slices):
                if ix == 0: continue
                if trend_decrease(sample):
                    ret = True
                    break
        return ret

    def trend_valley(diff_zrs):
        ret = False
        slices = slice_diffs(diff_zrs)
        if trend_decrease(slices[0]):
            for ix, sample in enumerate(slices):
                if ix == 0: continue
                if trend_increase(sample):
                    ret = True
                    break
        return ret
    
    shape = 0
    if trend_increase(diff_zrs):
        shape = 1
    elif trend_decrease(diff_zrs):
        shape = 2
    elif trend_hill(diff_zrs):
        shape = 3
    elif trend_valley(diff_zrs):
        shape = 4
        
    return shape

def find_track_outliers_zr(track, labels, hits, find_all=False, truth=None, debug=False):
    """Use z/r data to identify outliers. If find_all is False, only the first
    outlier will be identified, otherwise all potential outliers will be returned."""
    def find_extreme_jump(diff_zrs, head_threshold=10, tail_threshold=20, filter_mean=True):
        """The idea is to find jumps in the z/r value that are much more
        extreme than usual, by default 10-20x larger than the mean jump.
        Those extremes are classified as potential outliers."""
        rem_ix = -1
        new_mean = diff_zrs.mean()
        if filter_mean:
            filtered_diffs = diff_zrs[np.where(diff_zrs < 4*new_mean)[0]]
            new_mean = filtered_diffs.mean()

        min_removal_factor = head_threshold
        for ix, diff in enumerate(diff_zrs):
            # for curved tracks, the end of the track can have larger
            # values, so increase the removal threshold in the last 3rd
            if ix > int(len(diff_zrs)*0.65):
                min_removal_factor = tail_threshold
            if diff > min_removal_factor*new_mean:
                if ix == 0:
                    # May need to pick better candidate here. Makes sense to remove
                    # the first element (0) though - if the second element (1) was an
                    # outlier, that should cause the next diff magnitude to be wrong too.
                    rem_ix = 0
                else:
                    rem_ix = ix + 1
                break
        return rem_ix
    
    def find_opposing_extreme_jump(diff_zrs, head_threshold=10, tail_threshold=20, reverse_opt=False, favour_1st_removal=False):
        """The idea is to look for two large jumps in opposing directions, with up
        to two points in between. The first large jump is likely the incorrect one."""
        rem_ix = []
        abs_diff_zrs = np.absolute(diff_zrs)

        # First try to filter out any extreme positive or negative values to get
        # a more accurate mean. In some cases, this can filter out all values,
        # so retry using the mean of the absolute values. If reverse slope
        # optimizations are enabled, just use the standard mean - the diffs
        # are unreliable, so not meaningful to remove outliers
        if reverse_opt:
            new_mean = diff_zrs.mean()
        else:
            mean_diff_zr = diff_zrs.mean()
            filtered_diffs = diff_zrs[np.where(abs_diff_zrs < 4*mean_diff_zr)[0]]
            if (filtered_diffs.shape[0] == 0):
                mean_diff_zr = abs_diff_zrs.mean()
                filtered_diffs = abs_diff_zrs[np.where(abs_diff_zrs < 4*mean_diff_zr)[0]]
                new_mean = filtered_diffs.mean()
            else:
                new_mean = filtered_diffs.mean()

        min_removal_factor = head_threshold
        first_jump_ix = -1
        first_jump_val = 0
        second_jump_ix = -1
        second_jump_val = 0
        second_jump_factor = 1
        for ix, diff in enumerate(diff_zrs):
            # for curved tracks, the end of the track can have larger
            # values, so increase the removal threshold in the last 3rd
            if ix > int(len(diff_zrs)*0.65):
                min_removal_factor = tail_threshold
            if abs(diff) > abs(min_removal_factor*new_mean*second_jump_factor):
                if first_jump_ix == -1:
                    first_jump_ix = ix
                    first_jump_val = diff
                    second_jump_factor = 0.8
                else:
                    second_jump_ix = ix
                    second_jump_val = diff
                    break
        if (first_jump_ix != -1) and (second_jump_ix != -1) and ((second_jump_ix - first_jump_ix) <= 3):
            # Compare the jumps and distance between jumps to determine
            # potential outliers
            if ((first_jump_val >= 0) and (second_jump_val <= 0)) or ((first_jump_val <= 0) and (second_jump_val >= 0)):
                abs_diff = abs(first_jump_val + second_jump_val)
                if first_jump_ix == 0 and first_jump_val < 0 and (abs(first_jump_val) > 10*abs(new_mean)) and (abs(first_jump_val) > abs(second_jump_val)*1.2):
                    rem_ix.append(0)
                elif favour_1st_removal and (abs_diff < 0.1*abs(first_jump_val)):
                    rem_ix.append(first_jump_ix+1)
                elif reverse_opt and first_jump_val < 0 and first_jump_ix == 1 and diff_zrs[0] < 0 and (abs(first_jump_val) > 10*abs(new_mean)) and diff_zrs[first_jump_ix+1] > 0:
                    rem_ix.append(0)
                    rem_ix.append(1)
                elif reverse_opt and first_jump_val < 0 and first_jump_ix == 1 and diff_zrs[0] > 0 and (abs(first_jump_val) > 10*abs(new_mean)) and diff_zrs[first_jump_ix+1] > 0:
                    rem_ix.append(1)
                elif (abs(second_jump_val) < abs(first_jump_val)) or (abs_diff < 0.02*abs(first_jump_val)):
                    for i in range(first_jump_ix, second_jump_ix):
                        rem_ix.append(i+1)
        return rem_ix

    def find_negative_extreme_jump(diff_zrs, zrs, mean_diffs, jump_threshold=3):
        """The idea is to look for the biggest negative jump in a trending positive slope."""
        most_negative = 0
        most_negative_ix = -1
        rem_ix = -1
        for ix, diff in enumerate(diff_zrs):
            if diff < most_negative:
                most_negative = diff
                if ix == 0:
                    # Should be a better way to choose between 0 or 1 as outlier.
                    # Using 0 yields a better score, and is safer - track extension
                    # can recover lost hits from the end
                    most_negative_ix = 0
                else:
                    most_negative_ix = ix + 1
        if (most_negative < 0) and (abs(most_negative) > abs(mean_diffs)*jump_threshold):
            # can have cases like 0.01, 0.01, 0.5, -0.4, 0.1. -0.4 looks like
            # the outlier, except it's likely that 0.5 was too big a jump.
            # look at neighbours to see which one is likely the outlier.
            if most_negative_ix > 1 and (diff_zrs[most_negative_ix-2] > 0) and (abs(diff_zrs[most_negative_ix-2]) > abs(diff_zrs[most_negative_ix-1])):
                most_negative_ix = most_negative_ix - 1
            # Determine which is the best outlier to remove when the first slope is extreme
            elif most_negative_ix == 0 and zrs[2] > zrs[0]:
                most_negative_ix = 1

            rem_ix = most_negative_ix

        return rem_ix

    def find_positive_extreme_jump(diff_zrs, zrs, mean_diffs, jump_threshold=3):
        """The idea is to look for the biggest positive jump in a trending negative slope."""
        most_positive = 0
        most_positive_ix = -1
        rem_ix = -1
        for ix, diff in enumerate(diff_zrs):
            if diff > most_positive:
                most_positive = diff
                if ix == 0:
                    # Default to 0, code below will switch to 1 if that looks like
                    # the real outlier.
                    most_positive_ix = 0
                else:
                    most_positive_ix = ix + 1
        if (most_positive > 0) and (most_positive > abs(mean_diffs)*jump_threshold):
            # can have cases like -0.01, -0.01, -0.5, 0.4, -0.1. 0.4 looks like
            # the outlier, except it's likely that -0.5 was too big a jump.
            # look at neighbours to see which one is likely the outlier.
            if most_positive_ix > 1 and (diff_zrs[most_positive_ix-2] < 0) and (abs(diff_zrs[most_positive_ix-2]) > abs(diff_zrs[most_positive_ix-1])):
                most_positive_ix = most_positive_ix - 1
            # Determine which is the best outlier to remove when the first slope is extreme
            elif most_positive_ix == 0 and (zrs[2] < zrs[0]):# and (zrs[2] < zrs[1]):
                most_positive_ix = 1

            rem_ix = most_positive_ix

        return rem_ix

    def find_simple_opposing_jump(diff_zrs, threshold=5):
        """The idea is to look for two large jumps in opposing directions.
        The first large jump is likely the incorrect one."""
        rem_ix = -1
        abs_diff_zrs = np.absolute(diff_zrs)
        new_mean = diff_zrs.mean()
        prev_diff = abs(new_mean)
        for ix, diff in enumerate(diff_zrs):
            if (abs(diff) > threshold*prev_diff) and (abs(diff) > abs(threshold*new_mean)) and (ix < len(diff_zrs)-1) and (abs(diff_zrs[ix+1]) > abs(threshold*new_mean)):
                if ((diff > 0 and diff_zrs[ix+1] < 0) or (diff < 0 and diff_zrs[ix+1] > 0)) and (abs(diff_zrs[ix+1]) < abs(diff)) and (abs(diff_zrs[ix+1])*2 > abs(diff)):
                    # We found our candidate for removal, we have two opposing jumps much larger than
                    # the mean, and of roughly equivalent size, but the second jump is smaller. If
                    # the first candidate is the first diff, choose element 0, it's safest (can
                    # always be extended again via track extension).
                    if ix == 0:
                        rem_ix = ix
                    else:
                        rem_ix = ix+1
                    break
            prev_diff = abs(diff)

        return rem_ix

    def find_biggest_opposing_jump(diff_zrs, threshold=5, prev_diff_threshold=5):
        """The idea is to look for two large jumps in opposing directions.
        The first large jump is likely the incorrect one."""
        rem_ix = -1
        new_mean = diff_zrs.mean()
        prev_diff = abs(new_mean)
        biggest_diff = 0
        biggest_ix = -1
        for ix, diff in enumerate(diff_zrs):
            if (abs(diff) > abs(biggest_diff)) and (abs(diff) > prev_diff_threshold*prev_diff) and (abs(diff) > abs(threshold*new_mean)) and (ix < len(diff_zrs)-1) and (abs(diff_zrs[ix+1]) > abs(threshold*new_mean)):
                diff2 = diff_zrs[ix+1]
                if ((diff > 0 and diff2 < 0) or (diff < 0 and diff2 > 0)):
                    # Make sure the jumps are comparable, they are both much larger than the mean,
                    # but make sure one is not more than double the other jump
                    if (abs(diff) > abs(diff2) and abs(diff2*2) > abs(diff)) or (abs(diff2) > abs(diff) and abs(diff*2) > abs(diff2)):
                        biggest_diff = diff
                        biggest_ix = ix
            prev_diff = abs(diff)

        if biggest_ix != -1:
            # We found our candidate for removal, we have two opposing jumps much larger than
            # the mean, and of roughly equivalent size. Try to pick the best candidate when
            # the 0th diff contains the suspected outlier.
            if biggest_ix == 0 and ((diff_zrs[1] > 0 and diff_zrs[2] > 0) or (diff_zrs[1] < 0 and diff_zrs[2] < 0)):
                rem_ix = 0
            else:
                rem_ix = biggest_ix+1

        return rem_ix
    
    def find_final_slope_too_large(diff_zrs, threshold=10):
        """The idea is to find a final slope that is much larger than the previous slopes."""
        rem_ix = -1
        if diff_zrs[-1] > 0 and diff_zrs[-2] > 0 and diff_zrs[-3] > 0 and diff_zrs[-4] > 0:
            if (diff_zrs[-1] > threshold*diff_zrs[-2]) and (diff_zrs[-1] > threshold*diff_zrs[-3]) and (diff_zrs[-1] > threshold*diff_zrs[-4]):
                rem_ix = len(diff_zrs)
        return rem_ix

    def find_first_slope_wrong(diff_zrs, mean, threshold=3):
        """The idea is that valleys should start with negative slopes, if it starts with
        a positive slope followed by several negative slopes, the initial slope is likely wrong."""
        rem_ix = -1
        if diff_zrs[0] < 0 and abs(diff_zrs[0]) > threshold*abs(mean) and abs(diff_zrs[0]) > abs(diff_zrs[1]) and diff_zrs[1] > 0 and diff_zrs[2] > 0 and diff_zrs[3] > 0:
            rem_ix = 0
        elif diff_zrs[0] > threshold*abs(mean) and abs(diff_zrs[0]) > abs(diff_zrs[1]) and diff_zrs[1] < 0 and diff_zrs[2] < 0 and diff_zrs[3] < 0:
            rem_ix = 0
        return rem_ix
    
    outlier_ix = []
    hit_ix = np.where(labels == track)[0]

    # Need at least 5 values to determine if any look like outliers
    if len(hit_ix) < 5:
        return outlier_ix

    df = hits.loc[hit_ix]        
    df = df.sort_values('z')
    hit_ix2 = df.index.values
    
    zr_values = df['zr'].values
    abs_zrs = np.absolute(zr_values)
    diff_zrs = np.diff(zr_values)
    abs_diff_zrs = np.absolute(diff_zrs)
    mean_diff_zr = diff_zrs.mean()
    median_zr = abs(np.median(zr_values))
    #count_outliers = 0

    # If all diffs < 5% of the median value, track seems good
    if np.all(abs_diff_zrs < (median_zr * 0.05)):
        return outlier_ix

    shape = classify_zr_shape(abs_zrs, diff_zrs)

    rem_stage = 0
    rem_ix = -1
    new_mean = mean_diff_zr # Should re-calculate this after removing the outlier.

    # Common checks, make sure slopes at beginning are consistent,
    # and that the final slope at the end looks reasonable
    # Note we only typically remove one outlier per track at a time.
    rem_ix = find_first_slope_wrong(diff_zrs, new_mean)
    if rem_ix != -1:
        rem_stage = 1
        outlier_ix.append(hit_ix2[rem_ix])
    if find_all or rem_ix == -1:
        rem_ix = find_final_slope_too_large(abs_diff_zrs, threshold=20)
        if rem_ix != -1:
            rem_stage = 2
            outlier_ix.append(hit_ix2[rem_ix])
        if find_all or rem_ix == -1:
            rem_ix = find_biggest_opposing_jump(diff_zrs, threshold=10, prev_diff_threshold=5)
            if rem_ix != -1:
                rem_stage = 3
                outlier_ix.append(hit_ix2[rem_ix])
            if find_all or rem_ix == -1:
                if len(np.where(diff_zrs < 0)[0]) > len(np.where(diff_zrs > 0)[0]):
                    ndiff_zrs = np.copy(diff_zrs) * -1
                else:
                    ndiff_zrs = diff_zrs
                rem_ixes = find_opposing_extreme_jump(ndiff_zrs, head_threshold=10, tail_threshold=20, reverse_opt=True, favour_1st_removal=True)
                if len(rem_ixes) > 0:
                    rem_stage = 4
                    rem_ix = rem_ixes[0]
                    for ix in rem_ixes:
                        outlier_ix.append(hit_ix2[ix])
                if find_all or rem_ix == -1:
                    rem_ix = find_extreme_jump(abs_diff_zrs, head_threshold=20, tail_threshold=30)
                    if rem_ix != -1:
                        rem_stage = 5
                        outlier_ix.append(hit_ix2[rem_ix])
                    if find_all or rem_ix == -1:
                        if shape == 1 or shape == 2:
                            # Trending positive (1) or negative (2) slope, most values are either +ve or -ve
                            test_mean = new_mean
                        else:
                            # Hill-shaped (3), valley-shaped (4), or random/undetermined shape (0)
                            # Range of +ve and -ve values, take mean of absolute values to ensure the +ve
                            # and -ve values do not cancel each other out and make the mean too small.
                            test_mean = abs_diff_zrs.mean()

                        if len(np.where(diff_zrs > 0)[0]) > len(np.where(diff_zrs < 0)[0]):
                            rem_ix = find_negative_extreme_jump(diff_zrs, zr_values, test_mean, jump_threshold=4)
                        else:
                            rem_ix = find_positive_extreme_jump(diff_zrs, zr_values, test_mean, jump_threshold=4)
                        if rem_ix != -1:
                            rem_stage = 6
                            outlier_ix.append(hit_ix2[rem_ix])
                        if find_all or rem_ix == -1:
                            test_mean = abs_diff_zrs.mean()
                            if len(np.where(diff_zrs > 0)[0]) > len(np.where(diff_zrs < 0)[0]):
                                rem_ix = find_positive_extreme_jump(diff_zrs, zr_values, test_mean, jump_threshold=8)
                            else:
                                rem_ix = find_negative_extreme_jump(diff_zrs, zr_values, test_mean, jump_threshold=8)
                            if rem_ix != -1:
                                rem_stage = 7
                                outlier_ix.append(hit_ix2[rem_ix])

    if rem_ix == -1:
        return outlier_ix
    
    #print(str(shape) + ', ' + str(rem_ix) + ', ' + str(new_mean) + ', ' + str(mean_diff_zr) + ', ' + str(diff_zrs))# + ', ' + str(abs_zrs))
    #print('ami: ' + str(allowed_min) + ', amx: ' + str(allowed_max) + ', all: ' + str(abs_zrs))
    #print(diff_zrs)
    #print(hit_ix2)
        
    if truth is not None:
        tdf = truth.loc[hit_ix]
        truth_count = coll.Counter(tdf.particle_id.values).most_common(2)
        truth_particle_id = truth_count[0][0]
        if truth_particle_id == 0 and len(truth_count) > 1:
            truth_particle_id = truth_count[1][0]
        truth_ix = []
        count_true = 0
        count_false = 0
        for ix in hit_ix2:
            truth_ix.append(truth.loc[ix].particle_id == truth_particle_id)
            if truth.loc[ix].particle_id == truth_particle_id:
                count_true = count_true + 1
            else:
                count_false = count_false + 1
        tt_ix = np.where(truth.particle_id.values == truth_particle_id)[0]
        #majority1 = (count_true >= count_false)
        #majority2 = (count_true >= int(len(tt_ix)/2))
        #print(str(len(hit_ix2)) + ' ' + str(majority1) + ' ' + str(majority2) + ', Truth length: ' + str(len(tt_ix)) + ', True: ' + str(count_true) + ', False: ' + str(count_false))
        #print(truth_ix)
        #if truth_ix[rem_ix] == False:
        #    print('AWESOME: ' + str(truth_ix))
        #else:
        if debug:# and truth_ix[rem_ix] == True:
            print(str(shape) + ', ' + str(rem_ix) + ', ' + str(new_mean) + ', ' + str(mean_diff_zr) + ', ' + str(diff_zrs) + ', ' + str(zr_values))
            print('CRAPPY:  ' + str(rem_stage) + ', ' + str(truth_ix))

    return list(np.unique(outlier_ix))


def remove_outliers_zr(labels, hits):
    """Remove hits from tracks where those hits appear to be outliers, based on evaluation
    of the hit z/r values."""
    labels = np.copy(labels)
    tracks = np.unique(labels)
    hits['z_abs'] = hits.z.abs()
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['zr'] = hits['z'] / hits['r']
    count_rem_zr_slope = 0
    for track in tracks:
        if track == 0:
            continue
        track_hits = np.where(labels == track)[0]
        if len(track_hits) > 4:
            outliers = find_track_outliers_zr(track, labels, hits)
            if len(outliers) > 0:
                count_rem_zr_slope = count_rem_zr_slope + len(outliers)
                for oix in outliers:
                    labels[oix] = 0
            
    print('zr outliers removed: ' + str(count_rem_zr_slope))

    return labels

def safe_outlier_removal(labels, hits, truth, debug=False):
    """For training events, only remove hits identified as outliers if they do not really
    belong to the track, based on ground truth data. This allows for evaluation of the
    z/r outlier detection algorithm - i.e. determine false positive rate."""
    labels = np.copy(labels)
    tracks = np.unique(labels)
    hits['z_abs'] = hits.z.abs()
    hits['r'] = np.sqrt(hits.x**2+hits.y**2)
    hits['zr'] = hits['z'] / hits['r']
    count_removed = 0
    count_not_removed = 0
    for track in tracks:
        if track == 0:
            continue
        track_hits = np.where(labels == track)[0]
        if len(track_hits) > 3:
            outlier_ix = find_track_outliers_zr(track, labels, hits, truth=truth, debug=debug)
            if len(outlier_ix) > 0:
                tdf = truth.loc[track_hits]
                truth_count = coll.Counter(tdf.particle_id.values).most_common(1)
                truth_particle_id = truth_count[0][0]
                for out_ix in outlier_ix:
                    if tdf.loc[out_ix].particle_id != truth_particle_id:
                        labels[out_ix] = 0
                        count_removed = count_removed + 1
                    else:
                        count_not_removed = count_not_removed + 1

    print('safe count_removed: ' + str(count_removed))
    print('safe count_not_removed: ' + str(count_not_removed))
    return labels


