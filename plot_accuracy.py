import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

ROOT_DIR = "training_stats/*.stats"
files = glob.glob(ROOT_DIR)

labels = ['Bidirectional LSTM', 'CNN + LSTM', 'GRU', 'SeqConv1', 'SeqConv2']

def plot_graph(type:str):
    fig, ax = plt.subplots()
    if type == "training":
        draw_accuracy(ax, type)
        label_y = ax.set_ylabel('Training accuracy', fontsize = 15)
    elif type == "validation":
        draw_accuracy(ax, type)
        label_y = ax.set_ylabel('Validation accuracy', fontsize = 15)
    legend = ax.legend(loc=4, shadow=True)
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(0,120, 10))
    label_x = ax.set_xlabel('Number of batches', fontsize = 15)
    plt.show()

def draw_accuracy(ax, type):
    i = 0
    for f in files:
        if type == "training":
            reader = np.loadtxt(f, dtype=float, delimiter=',', skiprows=1, usecols=(1,))
        elif type == "validation":
            reader = np.loadtxt(f, dtype=float, delimiter=',', skiprows=1, usecols=(3,)) 
        x = range(1, len(reader)+1)
        ax.plot(x, reader, label=labels[i])
        i+=1

if __name__ == "__main__":
    plot_graph(sys.argv[1])
