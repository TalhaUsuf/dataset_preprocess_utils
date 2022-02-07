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
 - Process the **csv** file generated, convert to one-hot encoded labels and apply *iterative-train-test split* to get train and test sets. Both *train* as well as *test* steps will be *balanced*. :balance_scale:
  
```bash
    python iterative_split.py -c dataset.csv -t train.csv -v valid.csv
```
![](./histogram.png)