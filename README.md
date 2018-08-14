# kaggle-trackml
Kaggle TrackML challenge 2018

**Ensembling Helix 42 - #12 Solution**

Why 42? That's the largest internal DBScan helix cluster ID we merged. If you include each z-shift as a separate model, we actually merge a total of 45 models. That's a lot of merging!

[You can find all our code in this github repository](https://github.com/jliamfinnie/kaggle-trackml.git)

**Non-mathematicians solution from second-time Kagglers**

Nicole and I (Liam Finnie) started this Kaggle competition because it sounded pretty cool, however without a strong math or physics background, we quickly found ourselves at a disadvantage. So, we did what we know - write lots of code! Hopefully at least some of this will prove useful to someone, even as an example of 'what not to do!'.

Our solution consists of many DBScan variants with different features, z-shifts, etc. For post-processing, we use heavy outlier removal (both hits and entire tracks) and track extension individually on each of the DBScan results. We then split each of the results into 3 categories - strong, medium, and weak - before merging them.

**DBScan results**

Many thanks to @Luis for providing us our base DBScan kernel. Nicole did most of the math work on our team to develop our clustering features. I can't do the math justice, so if you understand advanced helix math, check out `hits_clustering.py`, class `Clusterer`, method `dbscan()`. We used several of the features discussed in the forum such as z-shifts and sampled `r0` values, as well as some of our own tweaks. Our raw DBScan scores tended to mostly be in the range of 0.35 to 0.55.

**Outlier removal**

Outlier removal is tricky - it lowers your LB, however allows for much better merging later on. Approaches we used for outlier removal:
- use `z/r` to eliminate hits that are out-of-place
- look for hits with the exact same `z` value from the same `volume_id` and `layer_id`, remove one of them.
- calculate the slope between each pair of adjacent hits, remove hits whose slopes are very different.

The outlier removal code entry point is in `merge.py` in function `remove_outliers()`.

**Helix Track extension**

Many thanks to @Heng who provided an initial track extension prototype. From this base, we added:
- `z/r` to improve the KDTree clustering
- track scoring (length + quality of track) to determine when to steal hits from another track
- different number of KDTree neighbours, angle slices, etc.

The track extension code can be found in the `extension.py` file, function `do_all_track_extensions()`. This type of track extension typically gave us a boost of between 0.05 and 0.15 for a single DBScan model.

**Straight Track extension**

Some tracks are more 'straight' than 'helix-like' - we do straight-track extension for track fragments from volumes 7 or 9. To extend straight tracks, we:
- compute `z/r` for each hit
- if our track does not have an entry in the adjacent `layer_id`, calculate the expected `z/r` for that adjacent `layer_id`, and assign any found hits to our track
- try to merge with track fragments from an adjacent `volume_id`.
    
This type of track extension typically gave us a boost of between 0.01 and 0.02 for a single DBScan model. Code is in `straight_tracks.py`, function `extend_straight_tracks()`.

**Merging**

When merging clusters with different z-shifts, we found the order mattered a lot - for example, we could merge better with z-shifts in the order (-6, 3, -3, 6), than in the order (-6, -3, 3, 6).

For our final merge at the end, we split each DBScan cluster into 'strong', 'medium' and 'weak' components based on the consistency of the helix curvature. Strong tracks are merged first, then medium, and finally weak ones at the end, getting more conservative at each step.

The main problem with merging is how to tell whether two tracks are really the same, or should be separate? We tend to favour extending existing tracks when possible, but will create a new track if there is too little overlap with any existing track. Some rough pseudo-code for our merging heuristics:
```foreach new_track in new_tracks:
   if (no overlap with existing tracks)
     assign new_track to merged results
   elif (existing track is longer than new_track and includes all hits)
     do nothing
   else
     determine longest overlapping track
     if (longest overlapping track is track '0', i.e. unassigned hits)
       consider second longest track for extension
     if (too little overlap with existing longest overlapping track)
       assign non-outlier hits from new_track to merged results
     else
       extend longest track to include non-outlier hits from new_track
```

Our merging code is in `merge.py`, function `heuristic_merge_tracks()`. We found simple merging ('longest-track-wins') hurts scores when there are more than 2 or 3 models, our current merging code was able to merge about 20 different sets of DBScan cluster results well.


**Acknowledgement**

Thanks to all Kagglers sharing in this competition, notably @Luis for the initial DBScan kernel, @Heng for the track-extension code, and @Yuval, @CPMP, @johnhsweeney, the chemist @Grzegorz, and many others for good discussions and DBScan feature suggestions.

<br>

# Hits clustering

## Training/Test data
- under [input/train and input/test](https://www.kaggle.com/c/trackml-particle-identification/data)
- r0 samples used during DBScan clustering under `input/r0_list`, generated by `src/notebooks/generate_radii_samples.ipynb`

## Main Driver

- find helixes and display scores for training data for 2 events starting from event 3, `python hits_clustering.py --train 3 2`

- find helixes and generate submission for all test data, `python hits_clustering.py --test 0 125`
