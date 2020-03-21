import pandas as pd
import glob

# get list of train .txt files
PREFIX_PATH = "../groundTruth/"
train_file_paths = sorted(glob.glob(PREFIX_PATH + "*.txt"))

# get segment splits for train .txt files
train_segments_file = open('train_segments.txt', 'w')
for train_file_path in train_file_paths:
    # get list of sub_action labels
    print("Get segment split for file: ", train_file_path)
    text_file = open(train_file_path, "r")
    labels = text_file.read().split()
    text_file.close()

    # get tuples of start and end indices of contiguous segments
    labels_pd = pd.DataFrame({"sub_action": labels})
    segments = (labels_pd != labels_pd.shift()).cumsum()
    segments = segments[labels_pd["sub_action"] != "SIL"]  # ignore "SIL" segments
    segment_split_indices = list(segments.groupby("sub_action").apply(lambda x: (x.index[0], x.index[-1])))
    print(segment_split_indices)

    # convert into segment format given for test files
    segment_split = ""
    for (idx, segment_indices) in enumerate(segment_split_indices):
        # change to 1-based indexing
        start = segment_indices[0] + 1
        end = segment_indices[1] + 1

        if idx == 0:
            segment_split += str(start) + " " + str(end)
        else:
            segment_split += " " + str(end)

    # append line to file
    print(segment_split)
    train_segments_file.write("%s\n" % segment_split)

print("Finished getting segment splits for train .txt files")
train_segments_file.close()
