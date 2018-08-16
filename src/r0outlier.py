import numpy as np
import pandas as pd
import math
import collections as coll
from scipy.optimize import least_squares


def estimate_helix_r0(track_ix, hits):
    """Estimate the r0 value for the input track."""
    def estimate_r0(x, y):
        def residuals_xy(param, x, y):
            x0, y0 = param
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            d = r - r.mean()
            return d

        param = (x.mean(), y.mean())
        res_lsq = least_squares(residuals_xy, param, loss='soft_l1', f_scale=1.0, args=(x,y))
        x0,y0 = res_lsq.x
        r0 = np.sqrt((x-x0)**2 + (y-y0)**2).mean()
        return r0

    df = hits.iloc[track_ix]
    t = df[['x', 'y', 'z']].values
    t = t[np.argsort(np.fabs(t[:,2]))]

    x = t[:,0]
    y = t[:,1]
    r0 = estimate_r0(x, y)

    x = t[0:int(len(t)/2),0]
    y = t[0:int(len(t)/2),1]
    r1 = estimate_r0(x, y)

    x = t[int(len(t)/2):,0]
    y = t[int(len(t)/2):,1]
    r2 = estimate_r0(x, y)

    return (r0,r1,r2)


def is_horrible_track(track, labels, hits):
    """Check for common qualities of poor quality tracks, such as when layers are missed,
    if there are too many hits in a single layer, or there are hits with identical z values."""
    hit_ix = np.where(labels==track)[0]
    df = hits.iloc[hit_ix]
    df = df.sort_values('z')
    vols = df.volume_id.values
    lays = df.layer_id.values
    zs = df.z.values
    dupz_count = 0
    seen_vols = [0]
    seen_lays = [0]
    horrible = 0
    last_lay_count = 0
    for ix, vol in enumerate(vols):
        if vol != seen_vols[-1]:
            seen_lays = [lays[ix]]
            # Check if vol in seen_vols (i.e. go back and forth between volumes)
            # 8/14/0 (so-so, not too great though, too many false positives!)
            seen_vols.append(vol)
            last_lay_count = 1
        elif lays[ix] != seen_lays[-1]:
            if vol != 7 and vol != 9 and (lays[ix] != (seen_lays[-1] + 2)) and (lays[ix] != (seen_lays[-1] - 2)):
                horrible = 3
                break
            seen_lays.append(lays[ix])
            last_lay_count = 1
        else:
            last_lay_count = last_lay_count + 1
            # count==7: 3/0/0, count==6: 15/2/0, count==5: 28/8/0, count==4: 83/87/2
            if last_lay_count == 4:
                horrible = 1
                break
        if ix > 0 and zs[ix] == zs[ix-1] and vol == vols[ix-1] and lays[ix] == lays[ix-1]:
            dupz_count = dupz_count + 1
            # count==1: 30/17/0, count==2: 21/4/0, count==3: 12/2/0, count==4: 11/0/0
            if dupz_count == 1:
                horrible = 2
                break

    return horrible


def remove_badr0_tracks(labels, hits):
    """Remove tracks whose estimated r0 values suggest they are poor quality."""
    tracks = np.unique(labels)
    for track in tracks:
        if track == 0: continue
        tix = np.where(labels==track)[0]
        if len(tix) < 4:
            labels[tix] = 0
            continue
        (r0,r1,r2) = estimate_helix_r0(tix, hits)
        #print('ii: ' + str(ii) + ', r0: '+ str(r0))
        if r2 > r1:
            distance = r2 - r1
        else:
            distance = r1 - r2
        # distance > 210 can remove up to about 1000 tracks,
        # and reduces score by about 0.005 or so.
        if distance > 210:
            #total_tracks_removed = total_tracks_removed + 1
            #total_hits_removed = total_hits_removed + len(tix)
            labels[tix] = 0
        elif int(r0) >= 325:
            # >325 seems to find ratio of about 2/3 horrible tracks to 1/3 imperfect tracks
            # >350 is more conservative, only removes really bad tracks
            labels[tix] = 0
        elif is_horrible_track(tix, labels, hits):
            labels[tix] = 0

    return labels

def find_circle_curvature(d01x, d01y, d12x, d12y):
    """Inputs: Given 3 points - x0, x1, x2 - convert to 2 vectors:
    1: x1-x0, y1-y0      2: x2-x1, y2-y1
    """
    x01 = d01x
    y01 = d01y
    x12 = d12x
    y12 = d12y
    x02 = x01 + x12
    y02 = y01 + y12
    # length of the triangle sides
    a = (x12**2 + y12**2)**0.5 #np.sqrt(x12**2, y12**2)
    b = (x02**2 + y02**2)**0.5 #np.sqrt(x02**2, y02**2)
    c = (x01**2 + y01**2)**0.5 #np.sqrt(x01**2, y01**2)
    # 2 * (signed) area of the triangle
    k = (x02 * y01 - x01 * y02)
    # radius = product of side lengths / 4 times triangle area
    if a == 0 or b == 0 or c == 0:
        return 10 # just a random large curvature value to use to avoid div-by-0
    else:
        return (2 * k) / (a * b * c)

def find_track_curvature(track, labels, hits):
    """Estimate the helix track curvature, assuming the helix includes (x,y)=(0,0).
    3 curvatures are returned:
    curv02a: curvature using 0,0, the first hit, and the mid-point hit of the helix
    curv02b: curvature using 0,0, the mid-point hit, and the last hit of the helix
    curv02c: curvature using 0,0, the second hit, and the last hit of the helix.
    """
    trk_ix = np.where(labels == track)[0]
    if len(trk_ix) < 5:
        #print('Track too short: ' + str(track))
        return (1, 1, 1)
    df = hits.loc[trk_ix]
    df = df.sort_values('z_abs')
    x = df.x.values
    y = df.y.values
    mid = int(len(x)/2)
    # Use (0,0) as starting point for cleaner results
    d01xa = x[0]
    d01ya = y[0]
    d12xa = x[mid] - d01xa
    d12ya = y[mid] - d01ya
    
    d01xb = x[mid]
    d01yb = y[mid]
    d12xb = x[-1] - d01xb
    d12yb = y[-1] - d01yb

    d01xc = x[1]
    d01yc = y[1]
    d12xc = x[-1] - d01xc
    d12yc = y[-1] - d01yc

    curv02a = find_circle_curvature(d01xa, d01ya, d12xa, d12ya)
    curv02b = find_circle_curvature(d01xb, d01yb, d12xb, d12yb)
    curv02c = find_circle_curvature(d01xc, d01yc, d12xc, d12yc)
    #print(curv02a)
    #print(curv02b)
    #print(curv02c)
    return (curv02a, curv02b, curv02c)

def find_bad_curvature_tracks(labels, hits, aggressive):
    """Find tracks whose curvature suggests they are of poor quality."""
    rejects = []
    tracks = np.unique(labels)
    if aggressive:
        reject_ratio = 0.40
    else:
        reject_ratio = 0.70
    for track in tracks:
        if track == 0: continue
        (curv1, curv2, curv3) = find_track_curvature(track, labels, hits)
        if np.sign(curv1) != np.sign(curv2) or np.sign(curv1) != np.sign(curv3):
            rejects.append(track)
        else:
            c1 = min(abs(curv1), abs(curv2))
            c2 = max(abs(curv1), abs(curv2))
            ratio = 1.0 - c1/c2
            if ratio > reject_ratio:
                rejects.append(track)
            elif False and ratio < 0.1:
                hit_ix = np.where(labels == track)[0]
                if is_horrible_track(track, labels, hits):
                    rejects.append(track)
                elif len(hit_ix) > 20:
                    rejects.append(track)
    return rejects

def remove_bad_curvature_tracks(labels, hits, aggressive):
    """Remove tracks whose curvature suggests they are of poor quality."""
    hits['z_abs'] = hits.z.abs()
    rejects = find_bad_curvature_tracks(labels, hits, aggressive)
    #print('Removing ' + str(len(rejects)) + ' tracks with bad curvature.')
    for reject in rejects:
        labels[labels == reject] = 0
    return labels

def split_tracks_based_on_quality(labels, hits):
    """Split input tracks into 3 categories - strong, medium, weak.
    Splitting is determined mainly by how consistent the track helix curvature is.
    Splitting tracks and merging different categories of tracks separately, and
    using more conservative merging for tracks with lower quality, improved
    the LB score by about 0.02 to 0.03"""
    strong_tracks = []
    medium_tracks = []
    weak_tracks = []
    hits['z_abs'] = hits.z.abs()
    tracks, counts = np.unique(labels, return_counts=True)
    strong_labels = np.zeros_like(labels)
    medium_labels = np.zeros_like(labels)
    weak_labels = np.zeros_like(labels)
    for ix, track in enumerate(tracks):
        if track == 0: continue
        if counts[ix] < 5:
            if counts[ix] > 3:
                medium_tracks.append(track)
            # else, discard, too short.
            continue
        (curv1, curv2, curv3) = find_track_curvature(track, labels, hits)
        if np.sign(curv1) != np.sign(curv2) or np.sign(curv1) != np.sign(curv3):
            weak_tracks.append(track)
            continue

        c1 = min(abs(curv1), abs(curv2))
        c2 = max(abs(curv1), abs(curv2))
        c3 = abs(curv3)
        ratio = 1.0 - c1/c2
        if ratio > 0.50:
            weak_tracks.append(track)
        elif ratio < 0.2:
            if counts[ix] > 20 or is_horrible_track(track, labels, hits):
                medium_tracks.append(track)
            else:
                strong_tracks.append(track)
        else:
            medium_tracks.append(track)

    for track in strong_tracks:
        strong_labels[labels==track] = track
    for track in medium_tracks:
        medium_labels[labels==track] = track
    for track in weak_tracks:
        weak_labels[labels==track] = track

    return (strong_labels, medium_labels, weak_labels)