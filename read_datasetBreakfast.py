# -*- coding: utf-8 -*-
import os
import torch
import pickle
import numpy as np
import os.path

import sklearn
from sklearn.model_selection import train_test_split


RAW_TRAINING_DATA_FILE = 'raw_training_data.p'
UNSORTED_TRAINING_DATA_FILE = 'unsorted_training_data.p'
TRAINING_DATA_FILE = 'training_data.p'
VALIDATION_DATA_FILE = 'validation_data.p'
TESTING_DATA_FILE = 'testing_data.p'


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def load_data(split_load, actions_dict, GT_folder, DATA_folder, datatype='training', ):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1] # because first line is #bundle and last line is blank
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    if datatype == 'training':
        print("==================================================")
        print("CREATING RAW TRAINING DATA FILE")

        data_breakfast_train_file = open(RAW_TRAINING_DATA_FILE, 'wb')

        data_breakfast = []
        labels_breakfast = []

        # read content of train segment splits
        train_segments_file = open('training_segment.txt', 'r')
        segment_ids = train_segments_file.read().split('\n')[:-1]  # last line is blank

        for (idx, content) in enumerate(content_all):
            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]
            label_seq, length_seq = get_label_length_seq(curr_gt)

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            # load data into memory
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])
            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            labels_breakfast.append(label_curr_video)

            # dump (segment, label) for current file into pickle
            curr_segment_ids = segment_ids[idx].split()
            for i in range(len(curr_segment_ids) - 1):
                start_segment_idx = int(curr_segment_ids[i])
                end_segment_idx = int(curr_segment_ids[i + 1])

                curr_segment_frames = curr_data[start_segment_idx:end_segment_idx]
                curr_segment_label = label_curr_video[start_segment_idx]
                pickle.dump((torch.tensor(curr_segment_frames, dtype=torch.float64), curr_segment_label),
                            data_breakfast_train_file)

            print(f'[{idx}] {content} contents dumped')

        labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
        print("Finished loading the training data and labels!")
        return data_breakfast, labels_uniq

    if datatype == 'test':
        print("==================================================")
        print("CREATING TESTING DATA FILE")

        data_breakfast_test_file = open(TESTING_DATA_FILE, 'wb')

        data_breakfast = []

        # read content of test segment splits
        test_segments_file = open('test_segment.txt', 'r')
        segment_ids = test_segments_file.read().split('\n')[:-1]  # last line is blank

        for (idx, content) in enumerate(content_all):
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            # load data into memory
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))

            # dump (segment, label) for current file into pickle
            curr_segment_ids = segment_ids[idx].split()
            for i in range(len(curr_segment_ids) - 1):
                start_segment_idx = int(curr_segment_ids[i])
                end_segment_idx = int(curr_segment_ids[i + 1])

                curr_segment_frames = curr_data[start_segment_idx:end_segment_idx]
                pickle.dump(torch.tensor(curr_segment_frames, dtype=torch.float64), data_breakfast_test_file)

            print(f'[{idx}] {content} contents dumped')

        print("Finished loading the test data!")
        return data_breakfast


def get_label_bounds(data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(0, len(data_labels)):
        uniq_group, indc_group = get_label_length_seq(data_labels[kki])
        labels_uniq.append(uniq_group[1:-1])
        labels_uniq_loc.append(indc_group[1:-1])
    return labels_uniq, labels_uniq_loc


def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq


def get_maxpool_lstm_data(cData, indices):
    list_data = []
    for kkl in range(len(indices) - 1):
        cur_start = indices[kkl]
        cur_end = indices[kkl + 1]
        if cur_end > cur_start:
            list_data.append(torch.max(cData[cur_start:cur_end, :],
                                       0)[0].squeeze(0))
        else:
            list_data.append(torch.max(cData[cur_start:cur_end + 1, :],
                                       0)[0].squeeze(0))
    list_data = torch.stack(list_data)
    return list_data


def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict


def create_validation_data():
    print("==================================================")
    print("CREATING TRAINING AND VALIDATION DATA FILES")

    f = open(RAW_TRAINING_DATA_FILE, "rb")

    training_data_inputs = []
    training_data_true_outputs = []
    counter = 1
    while True:
        try:
            (segment, label) = pickle.load(f)
        
            if counter % 100 == 0:
                print(f"at sample: {counter}")
            training_data_inputs.append(segment)
            training_data_true_outputs.append(label)
            counter += 1
        except (EOFError):
            break

    f.close()

    X_train, X_val, y_train, y_val = train_test_split(training_data_inputs, 
                training_data_true_outputs, test_size=0.2, random_state=42)


    print(len(X_train))
    print(len(X_val))

    # Store training data
    training_out = open(UNSORTED_TRAINING_DATA_FILE, 'wb')

    counter = 1
    for i, segment  in enumerate(X_train):
        if counter % 100 == 0:
            print(f"dumping sample {counter} in {training_out}") 
        pickle.dump((segment, y_train[i]), training_out)
        counter += 1
    training_out.close()

    # Store validation data
    validation_out = open(VALIDATION_DATA_FILE, 'wb')

    counter = 1
    for i, segment in enumerate(X_val):
        if counter % 100 == 0:
            print(f"dumping sample {counter} in {validation_out}")
        pickle.dump((segment, y_val[i]), validation_out)
        counter += 1

    validation_out.close()


def sort_training_data():
    print("==================================================")
    print("CREATING TRAINING DATA FILE SORTED BY SEGMENT LENGTH")  

    f = open(UNSORTED_TRAINING_DATA_FILE, "rb")

    segments = []
    labels = []
    segment_lengths = []

    segment_idx  = 0
    while True:
        try:
            (segment, label) = pickle.load(f)

            if segment_idx % 300 == 0:
                print(f"at sample: {segment_idx }")

            segments.append(segment)
            labels.append(label)
            segment_lengths.append((segment_idx , len(segment)))

            segment_idx  += 1
        except (EOFError):
            break

    f.close()

    sorted_segment_lengths = sorted(segment_lengths, key=lambda tup: tup[1])

    # Store sorted training data
    training_out = open(TRAINING_DATA_FILE, 'wb')

    counter = 1
    for (idx, length) in sorted_segment_lengths:
        if counter % 300 == 0:
            print(f"dumping sample {counter} in {training_out}") 

        pickle.dump((segments[idx], labels[idx]), training_out)
        counter += 1

    training_out.close()

if __name__ == "__main__":
    COMP_PATH = ''
    train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split = os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder = os.path.join(COMP_PATH, 'groundTruth/')
    DATA_folder = os.path.join(COMP_PATH, 'data/')
    mapping_loc = os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

    actions_dict = read_mapping_dict(mapping_loc)


    split = 'training'
    data_feat, data_labels = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype=split)
    
    split = 'test'
    data_feat = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype=split)

    create_validation_data()

    sort_training_data()
