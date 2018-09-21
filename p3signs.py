import os
import pickle
import csv
import matplotlib
import numpy as np
import math
from collections import OrderedDict

from sklearn.utils import shuffle
import tensorflow as tf


#====================== GLOBALS =====================

# For testing only
g_doTests = True
g_doPlots = True

g_TrainingFilesDirIn = "./Assets/training/"

#----------------------- PlottingBackend_Switch()
def PlottingBackend_Switch(whichBackEnd):
    import matplotlib
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print("Switched to:", matplotlib.get_backend())

#PlottingBackend_Switch('QT4Agg')
import matplotlib.pyplot as plt

#--------------------------------- ReadTrainingSet
def ReadTrainingSet():
    pass

#--------------------------------- ReadTrainingSets
def ReadTrainingSets():
    pass

signLabelsCSVFileIn = g_TrainingFilesDirIn + "signnames.csv"
training_file = g_TrainingFilesDirIn + "train.p"
validation_file = g_TrainingFilesDirIn + "valid.p"
testing_file = g_TrainingFilesDirIn + "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Create dictionaries from the csv. Assumes line 0 header and unique indices in order
with open(signLabelsCSVFileIn, mode='r') as infile:
    infile.readline() # Skip header lin 0
    reader = csv.reader(infile)
    dictIDToLabel = OrderedDict( (int(row[0]), row[1]) for row in csv.reader(infile) )
    labelsStr = dictIDToLabel.values()
    labelsIDs = dictIDToLabel.keys()
    #dictLabelToIndex = OrderedDict( (row[1], int(row[0])) for row in csv.reader(infile) )

# Get full set of ID's and the corresponding (first) image indices for each of of the IDs
imageLabelIDs, imageIndices = np.unique(y_train, return_index=True)

###################################### PLOTS ###########################
###################################### PLOTS ###########################
###################################### PLOTS ###########################

#--------------------------------- Plot_LabeledSampleImages
def Plot_LabeledSampleImages():
    figsize = (12,18)
    plotNumCols=4
    plotNumRows=11

    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})

    for (ax, imageIndex) in zip(axes.flat, imageIndices):
        labelID = y_train[imageIndex]
        label = dictIDToLabel[labelID]
        title = "{}:'{}'".format(labelID, label, imageIndex)
        ax.set_title(title, fontsize=8)
        ax.imshow(X_train[imageIndex])

    plt.tight_layout()
    plt.show()

#--------------------------------- compute_normal_histograms
def Plot_NumTrainingImagesHistogram():
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Number of training images")
    a = np.random.randint(2, size=1000)
    b = np.random.randint(2, size=1000)
    bins=np.arange(n_classes+1)-0.5
    plt.hist((y_train, y_test, y_valid), bins = bins, label = ("train", "test", "valid"))
    plt.legend()
    plt.xticks(imageLabelIDs)
    ax = plt.gca()
    histLables = ["{} : {}".format(label, labelID) for label, labelID in zip(labelsStr, labelsIDs)]
    ax.set_xticklabels(histLables, rotation=45, rotation_mode="anchor", ha="right")
    fig.subplots_adjust(bottom=0.5)
    plt.show()

###################################### TESTS ###########################
###################################### TESTS ###########################
###################################### TESTS ###########################

#====================== Main() =====================
def Main():
    if g_doPlots:
        Plot_LabeledSampleImages()
        Plot_NumTrainingImagesHistogram()

#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main()