import numpy as np
import random

class DataSource:
    def __init__(self, data_file, batch_size):
        self.start = 0
        self.dataset = {}
        self.batch_size = batch_size
        with open(data_file, "r") as f:
            lines = f.readlines()
            split_lines = [line.strip("\n").split(' ', 1) for line in lines]

            lines = np.array([line[1] for line in split_lines])
            targ =[line[0] for line in split_lines]
            targets = [0 if t == '-1' else 1 for t in targ]
            targets = np.array(targets)

            self.dataset["input"] = lines
            self.dataset["target"] = targets

        self.size = len(self.dataset["input"])

    #def next_train_batch(self, batch_size):
    #    batch_inputs = []
    #    batch_targets = []
    #    for i in range(batch_size):
    #        line = self.f.readline()
    #        if len(line) == 0:
    #            f.seek(0, 0)
    #            line = f.readline()

    #        line = line.strip("\n").split()
    #        inputs = np.array([int(x) for x in line])
    #        target = np.append(inputs[1:], self.pad_idx)

    #        batch_inputs.append(inputs)
    #        batch_targets.append(target)

    #    return batch_inputs, batch_targets
    def make_cvset(self, x, y, percent=0.1):
        cross_validation_indices = np.array(random.sample(list(np.arange(len(y))), int(len(y) * percent) ))
        train_indices = np.array(list(set(np.arange(len(y))) - set(cross_validation_indices)))

        x_train, x_test= x[train_indices], x[cross_validation_indices]
        y_train, y_test = y[train_indices], y[cross_validation_indices]
        return x_train, y_train, x_test, y_test



    def train_bitches(self):
        batch_inputs = []
        batch_targets = []
        end = self.start + self.batch_size
        batch_inputs = self.dataset["input"][self.start:end]
        batch_targets = self.dataset["target"][self.start:end]

        self.start = end

        if (len(batch_inputs) < self.batch_size):
            rest = self.batch_size - len(batch_inputs)
            batch_inputs = np.append(batch_inputs, self.dataset["input"][:rest])
            batch_targets = np.append(batch_targets, self.dataset["target"][:rest])
            self.start = rest

        return batch_inputs, batch_targets
