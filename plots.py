import readData
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

##############################################################
#  PLOTTING THE BAR CHART FOR NR OF SAMPLES IN EACH CATEGORY #
##############################################################

# read in the data
data = readData.Dataset("part2_training_data.csv")
def print_data_split(true_labels):

    labels, counts = np.unique(true_labels, return_counts=True)
    # graphical display of the number of training samples in each class
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Class", fontsize=15)
    ax.set_ylabel("Frequency", fontsize=15)
    ax.set_ylim(0, np.max(counts) + np.max(counts) * 0.15)
    rects = ax.bar(labels, counts, align='center', width=0.5, color='#0504aa',
                   alpha=0.7)

    ax.set_xticks(np.arange(0, 1.5, 1))
    # label the bars
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('images/class_vs_frequency_part2.png')
    plt.show()


def print_confusion_matrix(labels, predicted):
    assert len(labels) == len(predicted)
    cm = metrics.confusion_matrix(labels, predicted)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.grid(False)
    # annotate with exact numbers in boxes
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[i, j], xy=(j, i),
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=15, color='orange')

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', labelsize=13)
    ax.set_yticks(range(len(labels)), minor=False)
    ax.set_xticks(range(len(labels)), minor=False)
    ax.set_xticklabels(labels, minor=False, rotation=45, ha='left')
    ax.set_yticklabels(predicted, minor=False)
    plt.xlabel('True Label', fontsize=18, labelpad=30)
    plt.ylabel('Predicted Label', fontsize=18)
    ax.set_ylim(10 - 0.5, -0.5)
    plt.rcParams["axes.edgecolor"] = "0.6"
    plt.rcParams["axes.grid"] = False
    plt.show()

    plt.title("Confusion Matrix Plot", pad=50, fontdict={'fontsize': 20})

