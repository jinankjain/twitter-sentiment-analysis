import os
from random import shuffle
import sys

if __name__ == "__main__":
    raw_data_dir = sys.argv[1].strip("/")
    new_data_dir = sys.argv[2].strip("/")

    if os.path.isfile(raw_data_dir+"/test_data.txt"):
        os.rename(raw_data_dir+"/test_data.txt", new_data_dir+"/test_data.txt")
    if os.path.isfile(raw_data_dir+"/samples_submission.csv"):
        os.rename(
            raw_data_dir+"/sample_submission.csv",
            new_data_dir+"/samples_submission.csv")

    # Merge small training set files (with pos and neg samples) and shuffle the
    # samples.
    with open(raw_data_dir+"/train_pos.txt") as f_pos, \
            open(raw_data_dir+"/train_neg.txt") as f_neg, \
            open(new_data_dir+"/small_train.txt", "w") as f_out:
        small_pos_lines = f_pos.readlines()
        small_data = list(zip([1]*len(small_pos_lines), small_pos_lines))

        small_neg_lines = f_neg.readlines()
        small_data += list(zip([-1]*len(small_neg_lines), small_neg_lines))

        shuffle(small_data)
        for label, sentence in small_data:
            f_out.write(str(label) + " " + sentence)

    # Merge full training set files (with pos and neg samples) and shuffle the
    # samples.
    with open(raw_data_dir+"/train_pos_full.txt") as f_pos, \
            open(raw_data_dir+"/train_neg_full.txt") as f_neg, \
            open(new_data_dir+"/full_train.txt", "w") as f_out:
        full_pos_lines = f_pos.readlines()
        full_data = list(zip([1]*len(full_pos_lines), full_pos_lines))

        full_neg_lines = f_neg.readlines()
        full_data += list(zip([-1]*len(full_neg_lines), full_neg_lines))

        shuffle(full_data)
        for label, sentence in full_data:
            f_out.write(str(label) + " " + sentence)
