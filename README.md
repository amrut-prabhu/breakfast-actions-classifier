# Breakfast Actions Classification

The task is to perform video action classification on the [Breakfast actions dataset](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/).
This dataset includes 1712 videos and shows activities related to breakfast preparation.

## Usage

- Run `read_datasetBreakfast.py` to store the frames and the corresponding labels from the training data into a `pickle` file. This outputs `training_data.dat`.
- Run `create_validation_data.py`. This splits data in `training.dat` into `trimmed_training_data.p` and `validation_data.p` using a 80-20 ratio.
- ~~Run `preprocessing/create_segment.py` to get the start indices of segments in the training data. This outputs `preprocessing/train_segments.txt`.~~  
Use given `training_segment.txt` and `test_segment.txt` files instead.

## Data

- `data`: The video data can be downloaded from [here](https://drive.google.com/drive/folders/1KtpuFYRGXByf_9ICPsCbGRBoR_hLsruh).
It contains I3D features that are computed for each frame. 

- `groundTruth/`: The actual action labels for the training data video frames.

- `splits/`: Train-test split of the videos. Also contains mapping from action IDs to action names.

- `training_data.dat`: Created by `read_datasetBreakfast.py`. Contains tuples of video data and action labels, written using the `pickle` library.

## References

[H. Kuehne, A. B. Arslan and T. Serre. The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities. CVPR, 2014.](https://serre-lab.clps.brown.edu/wp-content/uploads/2014/05/paper_cameraReady-2.pdf)

I3D features: [Carreira J, Zisserman A. Quo vadis, action recognition? a new model and the kinetics dataset. IEEE Conference on Computer Vision and Pattern Recognition. 2017](https://arxiv.org/pdf/1705.07750.pdf)

