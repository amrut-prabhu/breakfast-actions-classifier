import numpy as np
import pandas as pd
import glob

# Maximum size of segment after processing training files:  5791
# File with maximum sized segment:  ../groundTruth/P17_webcam01_P17_scrambledegg.txt
# Segment split of train file with maximum sized segment:  [(16, 196), (197, 645), (646, 1212),
# (1213, 1362), (1363, 1786), (1787, 7577), (7578, 7651), (7652, 7835)]
# Maximum size of segment after processing test files:  3954
# Index of file in segment.txt:  228
# Segment split of test file with maximum sized segment:  [41, 456, 694, 1057, 2249, 2483, 2855, 6809, 6966, 7448]

# get list of train .txt files
PREFIX_PATH = "../groundTruth/"
train_file_paths = sorted(glob.glob(PREFIX_PATH + "*.txt"))

# get segment splits for train .txt files
max_segment_size_train = -np.inf
max_segment_file_path_train = ""  # for debugging
max_segment_file_indices_train = None  # for debugging
for train_file_path in train_file_paths:
    print("Processing file: ", train_file_path)
    text_file = open(train_file_path, "r")
    labels = text_file.read().split()
    text_file.close()

    # get tuples of start and end indices of contiguous segments
    labels_pd = pd.DataFrame({"sub_action": labels})
    segments = (labels_pd != labels_pd.shift()).cumsum()
    segments = segments[labels_pd["sub_action"] != "SIL"]  # ignore "SIL" segments
    segment_split_indices = list(segments.groupby("sub_action").apply(lambda x: (x.index[0], x.index[-1])))

    # find max segment size
    for (idx, segment_indices) in enumerate(segment_split_indices):
        curr_segment_size = segment_indices[1] - segment_indices[0] + 1
        if curr_segment_size > max_segment_size_train:
            max_segment_size_train = curr_segment_size
            max_segment_file_path_train = train_file_path
            max_segment_file_indices_train = segment_split_indices

print("Maximum size of segment after processing training files: ", max_segment_size_train)
print("File with maximum sized segment: ", max_segment_file_path_train)
print("Segment split of train file with maximum sized segment: ", max_segment_file_indices_train)

# get segment splits for test files from segment.txt
max_segment_size_test = -np.inf
test_segments_file_path = "../segment.txt"
test_segments_file = open(test_segments_file_path, "r")
segment_splits = test_segments_file.read().split("\n")
max_test_segment_indices = None
max_test_segment_file_idx = 0  # for debugging
for (idx, segment_split) in enumerate(segment_splits):
    indices = [int(n) for n in segment_split.split()]
    for i in range(len(indices) - 1):
        curr_segment_size = 0

        # need to add 1 for finding the length of the first segment
        if i == 0:
            curr_segment_size = 1

        curr_segment_size += indices[i + 1] - indices[i]
        if curr_segment_size > max_segment_size_test:
            max_segment_size_test = curr_segment_size
            max_test_segment_indices = indices
            max_test_segment_file_idx = idx

print("Maximum size of segment after processing test files: ", max_segment_size_test)
print("Index of file in segment.txt: ", max_test_segment_file_idx)
print("Segment split of test file with maximum sized segment: ", max_test_segment_indices)
test_segments_file.close()
