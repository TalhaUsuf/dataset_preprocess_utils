- [What this repo. is for?](#what-this-repo-is-for)
- [What it can do?](#what-it-can-do)
- [How to run?](#how-to-run)
  - [For balanced split ⚖️](#for-balanced-split-️)
  - [For Dis-joint / mutually-exclusive split ✂️](#for-dis-joint--mutually-exclusive-split-️)


# What this repo. is for?

This repo. has **parallelized** :lightning: :fire: code utilities for preprocessing pipeline of any face recognition dataset.


# What it can do?

It does following:

 - assigns labels to identities in parallel, optimized fashion and makes a csv file with "image", "identity" and "label" columns    
 - face detection :panda_face:
 - face aligning (image is rotated + scaled to make line joining eyes, horizontal :traffic_light: then it is scaled to a standard face)
 - resizing of images to some size defined by the user :man-gesturing-ok:


# How to run?

 - Following command runs a parallel code to effectivaly assign unique labels to each identity (all identities are defined by unique folders enclosed into a single folder defined by `--d` flag). It generates *separate csv files for each identity* and then *combines* them into single file (defined by `--c` flag)

```bash
    python generate_csv.py  --c ./dataset.csv --d images
```
## For balanced split ⚖️
 - Process the **csv** file generated, convert to one-hot encoded labels and apply *iterative-train-test split* to get train and test sets. Both *train* as well as *test* steps will be *balanced*. :balance_scale:
  
```bash
    python iterative_split.py -c dataset.csv -t train.csv -v valid.csv
```
![](./histogram.png)

## For Dis-joint / mutually-exclusive split ✂️

 - Following script creates totally dis-joint datasets, here -s 0.20 means 20% identities will be used for validation and rest for training:
    ```bash
    python disjoint_split.py  -c dataset.csv -t train.csv -v valid.csv -s 0.20
    ```