import os
import math
import numpy as np

import matplotlib
import matplotlib.image as mpimg

g_doSupressPlots = False

#----------------------- PlottingBackend_Switch()
def PlottingBackend_Switch(whichBackEnd):
    import matplotlib
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print("Switched to:", matplotlib.get_backend())

#PlottingBackend_Switch('QT4Agg')
import matplotlib.pyplot as plt

#====================== GLOBALS =====================
def PredictionComparison(dsTestRaw, dsTrainRaw, topk, figtitle=None):
    if (g_doSupressPlots):
        return

    if (figtitle is None):
        figtitle = "Prediction comparison for '{}'".format(dsTestRaw.name)

    figSize = (9.5, 2 * dsTestRaw.count * 1.5)
    fig, axes = plt.subplots(dsTestRaw.count, 4, figsize=figSize, subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(figtitle, y=1.0, fontsize=16)

    #fig.subplots_adjust(hspace=0.5, wspace=0.2)
    matchRanksStr = ["1st Match", "2nd Match", "3rd Match"]

    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(dsTrainRaw.y, return_index=True)

    for imageIndex in range(dsTestRaw.count):
        labelID = dsTestRaw.y[imageIndex]
        labelStr = dsTestRaw.dictIDToLabel[labelID]
        labelStr = (labelStr[:24] +"...") if len(labelStr) > 27 else labelStr
        title = "{}:'{}'".format(labelID, labelStr, imageIndex)

        xLabel = "TestImg: {}".format(dsTestRaw.imageNames[imageIndex])
        ylabel = dsTestRaw.imageNames[imageIndex]


        ax = axes[imageIndex][0]
        ax.imshow(dsTestRaw.X[imageIndex], interpolation='sinc')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xLabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        for matchRank, matchingLabelID in enumerate(topk.indices[imageIndex]):

            matchScore = topk.values[imageIndex][matchRank]
            matchingImageIndex = imageIndices[matchingLabelID]
            matchingImage = dsTrainRaw.X[matchingImageIndex]
            matchingLabelStr = dsTestRaw.dictIDToLabel[matchingLabelID]

            isMatchCorrect = (matchingLabelID == dsTestRaw.y[imageIndex])
            matchColor = "green" if isMatchCorrect else "black"
            matchFontSize = 12 if isMatchCorrect else 9

            labelID = matchingLabelID
            labelStr = matchingLabelStr
            labelStr = (labelStr[:24] + "...") if len(labelStr) > 27 else labelStr
            title = "{}:'{}'".format(labelID, labelStr)
            xLabel = "{}: {:0.3}".format(matchRanksStr[matchRank], matchScore)
            ylabel = dsTrainRaw.imageNames[matchingImageIndex]

            ax = axes[imageIndex][matchRank + 1]
            ax.imshow(matchingImage, interpolation='sinc')
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xLabel, fontsize=matchFontSize, color=matchColor)
            ax.set_ylabel(ylabel, fontsize=9)

    plt.tight_layout(pad=0.75) # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    plt.show()

#--------------------------------- Plot_LabeledSampleImages
def LabeledSampleImages(dsIn, figtitle=None):
    if g_doSupressPlots:
        return

    if (figtitle is None):
        figtitle = "Sample of dataset '{}'".format(dsIn.name)

    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(dsIn.y, return_index=True)
    numImages = len(imageLabelIDs)
    plotNumCols= min(6, numImages)
    plotNumRows= int(math.ceil(numImages/plotNumCols))
    figsize = (2 * plotNumCols + 1.5, 2 * plotNumRows + 1.5 )

    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(figtitle, y=1.0, fontsize=16)

    for (ax, imageIndex) in zip(axes.flat, imageIndices):
        labelID = dsIn.y[imageIndex]
        labelStr = dsIn.dictIDToLabel[labelID]
        labelStr = (labelStr[:24] +"...") if len(labelStr) > 27 else labelStr
        title = "{}:'{}'".format(labelID, labelStr, imageIndex)
        ylabel = dsIn.imageNames[imageIndex]

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.imshow(dsIn.X[imageIndex], cmap='viridis', interpolation='sinc')

    plt.tight_layout() # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    #plt.ion()  # turns on interactive mode
    plt.show()

#--------------------------------- compute_normal_histograms
def NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw):
    if g_doSupressPlots:
        return

    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(dsTrainRaw.y, return_index=True)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Number of training images")
    bins=np.arange(dsTrainRaw.numLabels+1)-0.5
    plt.hist((dsTrainRaw.y, dsValidRaw.y, dsTestRaw.y), bins = bins, label = ("train", "test", "valid"))
    plt.legend()
    plt.xticks(imageLabelIDs)
    ax = plt.gca()
    histLables = ["{} : {}".format(label, labelID) for label, labelID in zip(dsTrainRaw.labelsStr, dsTrainRaw.labelsIDs)]
    ax.set_xticklabels(histLables, rotation=45, rotation_mode="anchor", ha="right")
    fig.subplots_adjust(bottom=0.5)
    #plt.ion()  # turns on interactive mode
    plt.show()

