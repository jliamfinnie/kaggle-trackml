# kaggle-trackml
[Kaggle TrackML challenge 2018](https://www.kaggle.com/c/trackml-particle-identification)

**Ensembling Helix 42 - #12 Solution**

Why 42? That's the largest internal DBScan helix cluster ID we merged. If you include each z-shift as a separate model, we actually merge a total of 45 models. That's a lot of merging!

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

**Track Quality Classification**

For our final merge at the end, we split each DBScan cluster into **strong**, **medium** and **weak** components based on the consistency of the helix curvature. Strong tracks are merged first, then medium, and finally weak ones at the end, getting more conservative at each step.

This classification and merging approach gave us a boost of between 0.03 and 0.04, compared to merging all tracks at once. Code is in `r0outlier.py`, function `split_tracks_based_on_quality()`.

**Merging**

When merging clusters with different z-shifts, we found the order mattered a lot - for example, we could merge better with bigger jumps between successive z-shifts, i.e.the order (-6, 3, -3, 6) works better than (-6, -3, 3, 6).

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

## Generate the hidden feature - helix radii as input 

* I find the most tricky part doing clustering in this competition is simulating the range of helix radii. We've tried different distributions, such as linear distribution, Gaussian distribution, but the most effective way is to generate real helix radii from the train data. I stole @Heng's code from [this post](https://www.kaggle.com/c/trackml-particle-identification/discussion/57643).
 
* You can find [my notebook](https://github.com/nicolefinnie/kaggle-trackml/blob/master/src/notebooks/generate_radii_samples.ipynb) which generates helix radii from the train data or you can download pre-generated radii named after their event id [here](https://github.com/nicolefinnie/kaggle-trackml/tree/master/input/r0_list)

## Helix unrolling function

### z-axis centered tracks, tracks crossing (x,y) = (0,0)
* The closet approach `D0 = 0`, the tracks start from close to `(0,0,z)`.

![helix unrolling](https://github.com/nicolefinnie/kaggle-trackml/blob/master/images/helix_unrolling.png)
* Accurate version - reach `score=0.5` within 1 minute with 40 radius samples. To get a higher accuracy, you need to run more radius samples and it can take much longer.

```
# The track can go in either direction, and theta0 should be constant for all hits 
# on the same track in a perfect helix form.

  dfh['cos_theta'] = dfh.r/2/r0

  if ii < r0_list.shape[0]/2:
      dfh['theta0'] = dfh['phi'] - np.arccos(dfh['cos_theta'])
  else:
      dfh['theta0'] = dfh['phi'] + np.arccos(dfh['cos_theta'])

```

* Self-made version - reach `score=0.5` in 2 minutes but it rarely go above 0.5 since the unrolling function is not accurate. The reason why we use this self-made version is that it can find different tracks for later merging, which is good.

```
# This tries to find possible theta0 using an approximation function
# ii from -120 to 120

   STEPRR = 0.03
   rr = dfh.r/1000
   dfh['theta0'] = dfh['phi'] + (rr + STEPRR*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi


```

#### Main features that are constant in a perfect helix form
* `sin(theta0)`
* `cos(theta0)`
* `(z-z0)/arc`

* The problem is `z/arc` is still uneven in the magnetic field, so I've been trying to improve this problem by using following features in different models. Other Kagglers definitely have a more accurate equation. 
* `log(1 + abs((z-z0)/arc))*sign(z)`
* `(z-z0)/arc*sqrt(sin(arctan2(r,z-z0)))` I use this square root sine function as a correction term for the azimuthal angle on the x-y plane projection

#### Main features that are often constant
* `(z-z0)/r` where `r` is the Euclidean distance from the hit to the origin.
* `log(1 + abs((z-z0)/r))*sign(z)` is an approach to get z values closer to the origin to improve the problem with uneven `z/r` values

#### Side features
* Those are often not constant but we can find different tracks using them with small weights when we cluster
* `x/d` where `d` is the Eucliean distance from the hit to the origin in 3D `(x**2+y**2+z**3)**0.5`
* `y/d`
* `arctan2(z-r0,r)`
* `px, py` in my code: `-r*cos(theta0)*cos(phi)-r*sin(theta0)*sin(phi)` and `-r*cos(theta0)*sin(phi)+r*sin(theta0)*cos(phi)`. I happened to find this feature that can find the seeds of non-z-axis centered tracks and we can extend the tracks using the found seeds. 

### Non-z-axis centered tracks
* Skip this part since we didn't have time to implement it. Add the closest approach `D0` to your equations, you can find full equations with `D0` and `D1` from [Full helix equations for ATLAS](http://www.hep.ucl.ac.uk/atlas/atlantis/files/helix_equations_1.pdf)


## LSTM approach for track fitting
* My [notebook](https://github.com/nicolefinnie/kaggle-trackml/blob/master/src/notebooks/train_LSTM.ipynb) including visualization
* We take first five hits as a seeded track and predict next 5 hits. I believe this has its potential as a track validation tool if right features were trained. I shared the detail in [this post](https://www.kaggle.com/c/trackml-particle-identification/discussion/60455#352645)

## pointNet approach 
* PointNet is a lightweight CNN that can be used for pixel level classification(segementation) or classification. I put experimental code [here](https://github.com/nicolefinnie/kaggle-trackml/blob/master/src/train_pointnet.py), the challenge is to generate the right train data.


## Background knowledge 

* [Very good slides for beginners](http://ific.uv.es/~nebot/IDPASC/Material/Tracking-Vertexing/Tracking-Vertexing-Slides.pdf)

* [Lecture of particles tracking](http://www.physics.iitm.ac.in/~sercehep2013/track2_Gagan_Mohanty.pdf)


* [Full helix equations for ATLAS](http://www.hep.ucl.ac.uk/atlas/atlantis/files/helix_equations_1.pdf) - All equations you need!


* [Diplom thesis](http://physik.uibk.ac.at/hephy/theses/dipl_as.pdf) of Andreas Salzburger (Wow, he started in this field as a CERN student already in 2001 :stuck_out_tongue_closed_eyes: )

* [Doctor thesis](http://physik.uibk.ac.at/hephy/theses/diss_as.pdf) of Andreas Salzburger

* [CERN tracking software Acts](https://gitlab.cern.ch/acts/acts-core) - Sadly, we didn't have time to explore it :) 



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
