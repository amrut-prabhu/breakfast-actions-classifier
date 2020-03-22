# Breakfast Actions Classification

The task is to perform video action classification on the [Breakfast actions dataset](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).
This dataset includes 1712 videos and shows activities related to breakfast preparation.

## Usage

### Data processing

Run `read_datasetBreakfast.py` to process the raw i3D data files. It stores the segments (containing i3D frames) and the corresponding labels into binary `pickle` files. 

```sh
python read_datasetBreakfast.py
```

Outputs:
- `raw_training_data.p`: Training data with 7075 `(segment, label)` tuples.
- `unsorted_training_data.p`: Training data containing 80% of `raw_training_data.p` (5660 samples).
- `training_data.p`: Final training dataset with `(segment, label)` sorted in increasing order of segment lengths.
- `validation_data.p`: Validation data with `(segment, label)` containing 20% of `raw_training_data.p` (1415 samples).
- `testing_data.p`: Test data with only segments and no labels (1536, but should be 1284).

## Data

- `data`: The video data can be downloaded from [here](https://drive.google.com/drive/folders/1KtpuFYRGXByf_9ICPsCbGRBoR_hLsruh).
It contains I3D features that are computed for each frame. 

- `groundTruth/`: The actual action labels for the training data video frames.

- `splits/`: Train-test split of the videos. Also contains mapping from action IDs to action names.

- `training_data.dat`: Created by `read_datasetBreakfast.py`. Contains tuples of video data and action labels, written using the `pickle` library.

## References

[H. Kuehne, A. B. Arslan and T. Serre. The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities. CVPR, 2014.](https://serre-lab.clps.brown.edu/wp-content/uploads/2014/05/paper_cameraReady-2.pdf)

I3D features: [Carreira J, Zisserman A. Quo vadis, action recognition? a new model and the kinetics dataset. IEEE Conference on Computer Vision and Pattern Recognition. 2017](https://arxiv.org/pdf/1705.07750.pdf)

