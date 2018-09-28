import os
import math
import numpy as np

import matplotlib
import matplotlib.image as mpimg
import tensorflow as tf
g_doShowPlots = False

#----------------------- PlottingBackend_Switch()
def PlottingBackend_Switch(whichBackEnd):
    import matplotlib
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print("Switched to:", matplotlib.get_backend())

#PlottingBackend_Switch('TkAgg') # inline notebook GTK GTKAgg GTKCairo GTK3Agg GTK3Cairo Qt4Agg Qt5Agg TkAgg WX WXAgg Agg Cairo GDK PS PDF SVG
import matplotlib.pyplot as plt
#plt.ioff()  # turns on interactive mode

#====================== GLOBALS =====================
def PredictionComparison(dsTestRaw, dsTrainRaw, topk, figtitle=None):
    if (not g_doShowPlots):
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

    sess = tf.Session()

    for imageIndex in range(dsTestRaw.count):
        labelID = dsTestRaw.y[imageIndex]
        labelStr = dsTestRaw.dictIDToLabel[labelID]
        labelStr = (labelStr[:24] +"...") if len(labelStr) > 27 else labelStr
        title = "{}:'{}'".format(labelID, labelStr, imageIndex)

        xLabel = "TestImg: {}".format(dsTestRaw.imageNames[imageIndex])
        ylabel = dsTestRaw.imageNames[imageIndex]

        # Extract ndarray from tensor, if needed
        imgIn = dsTestRaw.X[imageIndex]
        if (isinstance(imgIn, tf.Tensor)):
            imgOut = sess.run(imgIn)
        else:
            imgOut = imgIn

        ax = axes[imageIndex][0]
        ax.imshow(imgOut, interpolation='sinc') # dsTestRaw.X[imageIndex]
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
            matchFontSize = 14 if isMatchCorrect else 9

            labelID = matchingLabelID
            labelStr = matchingLabelStr
            labelStr = (labelStr[:24] + "...") if len(labelStr) > 27 else labelStr
            title = "{}:'{}'".format(labelID, labelStr)
            xLabel = "{}: {:0.3}".format(matchRanksStr[matchRank], matchScore)
            ylabel = dsTrainRaw.imageNames[matchingImageIndex]

            # Extract ndarray from tensor, if needed
            imgIn = matchingImage
            if (isinstance(imgIn, tf.Tensor)):
                imgOut = sess.run(imgIn)
            else:
                imgOut = imgIn

            ax = axes[imageIndex][matchRank + 1]
            ax.imshow(imgOut, interpolation='sinc')
            ax.set_title(title, fontsize=matchFontSize, color=matchColor)
            ax.set_xlabel(xLabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)

    plt.tight_layout(pad=0.75) # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    plt.show()

#--------------------------------- Plot_LabeledSampleImages
def LabeledSampleImages(dsIn, figtitle=None):
    if not g_doShowPlots:
        return

    if (figtitle is None):
        figtitle = "Sample of dataset '{}'".format(dsIn.name)

    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(dsIn.y, return_index=True)
    numImages = len(imageLabelIDs)
    #if (numImages == 0):
    #    return

    plotNumCols= min(6, numImages)
    plotNumRows= int(math.ceil(numImages/plotNumCols))
    figsize = (2 * plotNumCols + 1.5, 2 * plotNumRows + 1.5 )

    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(figtitle, y=1.0, fontsize=16)

    sess = tf.Session()

    # Convert tensor to ndarray if needed
    if (isinstance(dsIn.X, tf.Tensor)):
        print("sess.run...", end='', flush=True)
        ndarrayX = sess.run(dsIn.X)
        print("done.")
    else:
        ndarrayX = dsIn.X

    if (isinstance(axes,np.ndarray)):
        axes = axes.flat
    else:
        axes = [axes]

    for (ax, imageIndex) in zip(axes, imageIndices):
        labelID = dsIn.y[imageIndex]
        labelStr = dsIn.dictIDToLabel[labelID]
        labelStr = (labelStr[:24] +"...") if len(labelStr) > 27 else labelStr
        title = "{}:'{}'".format(labelID, labelStr, imageIndex)
        ylabel = dsIn.imageNames[imageIndex]

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)

        imgOut = ndarrayX[imageIndex]
        ax.imshow(imgOut, cmap='viridis', interpolation='sinc') #dsIn.X[imageIndex]

    sess.close()

    plt.tight_layout() # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    #plt.ion()  # turns on interactive mode
    plt.show()

#--------------------------------- PlotListOfDataSets
def SingleImage(imgIn, title="TestImg"):
    figsize = (8, 6)

    fig = plt.figure(figsize=(15, 5))
    ax = plt.gca()
    #plt.imshow(imgIn, interpolation='sinc')  # interpolation='sinc', vmin=0.0, vmax=255.0, cmap="viridis"
    ax.imshow(imgIn, interpolation='sinc', )  # interpolation='sinc', vmin=0.0, vmax=255.0, cmap="viridis"
    ax.set_title(title, fontsize=12)
    plt.tight_layout(pad=-0.4) # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    plt.show()


#--------------------------------- PlotListOfDataSets
def PlotListOfDataSets(dataSetsList, figtitle=None):
    if not g_doShowPlots:
        return
    #plt.ioff()

    numDS = len(dataSetsList)
    numImagesPerDS = 3
    if (figtitle is None):
        figtitle = "{}x{} data set image samples".format(numDS, numImagesPerDS)

    print("Plotting list of datasets: {}...".format(figtitle))

    dsMetaRows = 2
    plotNumCols= int(math.ceil(numDS/dsMetaRows))
    plotNumRows= numImagesPerDS * dsMetaRows

    figsize = (1 * plotNumCols + 1.5, 1 * plotNumRows + 1.5 )

    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(figtitle, y=1.0, fontsize=16)

    sess = tf.Session()

    for (dsIndex, dsIn) in enumerate(dataSetsList):
        # Convert tensor to ndarray if needed
        if (isinstance(dsIn.X, tf.Tensor)):
            print("sess.run({})".format(dsIndex), end='\r', flush=True)
            ndarrayX = sess.run(dsIn.X)
        else:
            ndarrayX = dsIn.X

        for imageIndex in range( min(dsIn.count, numImagesPerDS)):
            print("\rds({:02}), img({})".format(dsIndex, imageIndex), end='', flush=True)
            labelID = dsIn.y[imageIndex]
            labelStr = dsIn.dictIDToLabel[labelID]
            labelStr = (labelStr[:10] +"...") if len(labelStr) > 13 else labelStr
            title = "{}:'{}'".format(labelID, labelStr, imageIndex)
            ylabel = dsIn.imageNames[imageIndex]

            imgOut = ndarrayX[imageIndex]

            colIndex = dsIndex % plotNumCols
            rowIndex = imageIndex + (numImagesPerDS * (dsIndex//plotNumCols))
            ax = axes[rowIndex][colIndex]
            ax.imshow(imgOut, interpolation='sinc',)#interpolation='sinc', vmin=0.0, vmax=255.0, cmap="viridis"
            ax.set_title(title, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)

    sess.close()

    plt.tight_layout(pad=-0.4) # (pad=-0.75 w_pad=0.5, h_pad=1.0)
    plt.show()

#--------------------------------- compute_normal_histograms
def NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw):
    if not g_doShowPlots:
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

