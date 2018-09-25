import os
import glob
import pickle
import csv
import math
import time
import re
from collections import OrderedDict
from collections import namedtuple
import copy
import matplotlib
import numpy as np
import matplotlib.image
import matplotlib.image as mpimg

import sklearn.utils
import tensorflow as tf
from tensorflow.contrib.layers import flatten

#----------------------- PlottingBackend_Switch()
def PlottingBackend_Switch(whichBackEnd):
    import matplotlib
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print("Switched to:", matplotlib.get_backend())

#PlottingBackend_Switch('QT4Agg')
import matplotlib.pyplot as plt

#----------------------- checkGPU()
def checkGPU():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
#checkGPU()

#====================== GLOBALS =====================
# Lower the verbosity of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, or FATAL

# For debug testing only
g_doTests = False
g_doSupressPlots = False
g_figSize = (12,9)
g_truncatedTrainingSetSize = 0 # Default: 0 (Use full training sets). Truncation is for speedup of debug cycle

g_doTrainModel = True
g_doSaveModel = True
g_doLoadModel = False

g_NUM_EPOCHS = 22
g_BATCH_SIZE = 64
g_TRAINRATE = 0.001

g_doConvertGray = False

g_TrainingFilesDirIn = "./Assets/training/"
g_finalTestFilesDirIn = "./Assets/finalTest/"
g_sessionSavesDir = "./Assets/tfSessionSaves/"
g_sessionSavesName = g_sessionSavesDir + "lenet"

signLabelsCSVFileIn = g_TrainingFilesDirIn + "signnames.csv"
training_file = g_TrainingFilesDirIn + "train.p"
validation_file = g_TrainingFilesDirIn + "valid.p"
testing_file = g_TrainingFilesDirIn + "test.p"

g_trainSets = type("DatasetContainer", (object,), {})
g_tph = type("tfPlaceHolders", (object,), {})

#====================== CTrainingDataSet() =====================
class CTrainingDataSet():
    # ----------------------- ctor
    def __init__(self, name="NoName", pickleFileNameIn = None, dictIDToLabel = None):
        self.X = None
        self.y = None
        self.numLabels = None
        self.count = None
        self.name = name
        self.SetLabelDict(dictIDToLabel)
        self.pickleFileNameIn = pickleFileNameIn

        if (not pickleFileNameIn is None):
            self.ReadPickleFile(pickleFileNameIn)

    #--------------------------------- SetLabelDict
    def SetLabelDict(self, dictIDToLabel):
        self.dictIDToLabel = dictIDToLabel
        if (self.dictIDToLabel is None):
            self.labelsStr = None
            self.labelsIDs = None
            self.numLabels = None
        else:
            self.labelsStr = self.dictIDToLabel.values()
            self.labelsIDs = self.dictIDToLabel.keys()
            self.numLabels = len(self.labelsIDs)

    #--------------------------------- ReadPickleFile
    def ReadPickleFile(self, pickleFileNameIn):
        with open(pickleFileNameIn, mode='rb') as f:
            dictDS = pickle.load(f)
            self.X, self.y = dictDS['features'], dictDS['labels']
            self.count = len(self.X)
            self.pickleFileNameIn = pickleFileNameIn

    #--------------------------------- GetXFeatureShape
    def GetXFeatureShape(self):
        return self.X[0].shape

    #--------------------------------- GetXFeatureShape
    def GetDSNumLabels(self):
        return len(set(self.y))

    #--------------------------------- DebugTruncate
    def DebugTruncate(self, truncSize):
        self.X = self.X[:truncSize]
        self.y = self.y[:truncSize]
        self.count = len(self.X)


#====================== CImageRecord() =====================
class CImageRec():
    # ----------------------- ctor
    def __init__(self, fileNameInFQ=None, labelId=None, labelStr=None):
        self.fileNameBase = os.path.basename(fileNameInFQ)
        self.fileNameFQ = fileNameInFQ
        self.labelId = labelId
        self.labelStr = labelStr
        self.img = None

    # ----------------------- GetImage()
    def GetImage(self):
        if self.img is None:
            self.img = mpimg.imread(self.fileNameFQ)
        return(self.img)

#--------------------------------- LoadTestImages
def LoadTestImages(dirIn, dictIDToLabel):
    imageRecs = []
    fileExt = '*.jpg'
    fileSpec = dirIn + fileExt
    print("\nReading Extra test sets: {}".format(fileSpec))

    fileNamesIn = glob.glob(fileSpec)

    reExtractLabelNum = "(\d\d)\."

    for fileNameIn in fileNamesIn:
        labelId = labelIdStr = None
        reResult = re.search(reExtractLabelNum, fileNameIn)
        if reResult:
            labelId = int(reResult.groups()[0])
            labelStr = dictIDToLabel[labelId]

        fileNameInFQ = os.path.abspath(fileNameIn)
        imageRec = CImageRec(fileNameInFQ, labelId, labelStr)
        imageRecs.append(imageRec)

        recStr = "{:<10} type {:02}: {}".format(imageRec.fileNameBase, imageRec.labelId, imageRec.labelStr)
        print(recStr)

    return(imageRecs)

#--------------------------------- GetNumRawInputChannels
def GetNumRawInputChannels():
    numRawInputChannels = 1 if g_doConvertGray else 3
    return numRawInputChannels

#--------------------------------- ReadTrainingSets
def ReadTrainingSets(training_file, validation_file, testing_file, dictIDToLabel=None):
    global g_trainSets

    print("\nReading Training sets...")
    dsTrainRaw = CTrainingDataSet(name="TrainRaw", pickleFileNameIn = training_file, dictIDToLabel = dictIDToLabel)
    dsValidRaw = CTrainingDataSet(name="ValidRaw", pickleFileNameIn = validation_file, dictIDToLabel = dictIDToLabel)
    dsTestRaw  = CTrainingDataSet(name="TestRaw", pickleFileNameIn = testing_file, dictIDToLabel = dictIDToLabel)

    # TODO: How many unique classes/labels there are in the dataset.
    g_trainSets.n_classes = dsTrainRaw.GetDSNumLabels()

    if g_truncatedTrainingSetSize > 0:
        print("*************** WARNING: Debug data sets truncated to size {}! *****************".format(g_truncatedTrainingSetSize))
        dsTrainRaw.DebugTruncate(g_truncatedTrainingSetSize)
        dsValidRaw.DebugTruncate(g_truncatedTrainingSetSize)
        dsTestRaw.DebugTruncate(g_truncatedTrainingSetSize)

    print("Raw training set size (train, validation, test) =({}, {}, {})".format(dsTrainRaw.count, dsValidRaw.count, dsTestRaw.count))
    print("Image data shape =", dsTrainRaw.GetXFeatureShape())
    print("Number of classes =", g_trainSets.n_classes)

    return dsTrainRaw, dsValidRaw, dsTestRaw

#--------------------------------- ReadLabelDict
def ReadLabelDict(signLabelsCSVFileIn):
    '''
    Create dictionaries from the csv. Assumes line 0 header and unique indices in order
    '''
    global g_trainSets

    with open(signLabelsCSVFileIn, mode='r') as infile:
        infile.readline() # Skip header line 0
        reader = csv.reader(infile)
        dictIDToLabel = OrderedDict( (int(row[0]), row[1]) for row in csv.reader(infile) )
    return dictIDToLabel

#--------------------------------- ShuffleDSInPlace
def ShuffleDSInPlace(dsIn):
    dsIn.X, dsIn.y = sklearn.utils.shuffle(dsIn.X, dsIn.y)

#--------------------------------- GrayscaleNormalize
def GrayscaleNormalize(Xin):
    dsInfloat = tf.cast(Xin, dtype=tf.float32)

    if g_doConvertGray:
        dsInfloat = tf.image.rgb_to_grayscale(dsInfloat)

    dsInfloat = tf.map_fn(lambda img: tf.image.per_image_standardization(img),  dsInfloat, dtype=tf.float32)

    Xout = []
    with tf.Session() as sess:
        Xout = dsInfloat.eval()

    return Xout

# --------------------------------- GrayscaleNormalizeRawDataSets
def GrayscaleNormalizeRawDataSets(listDSRaw):
    listDSNorm = []

    for dsRaw in listDSRaw:
        dsNorm = copy.copy(dsRaw)
        dsNorm.X = GrayscaleNormalize(dsRaw.X)
        dsNorm.name += "Norm"
        listDSNorm.append(dsNorm)

    return listDSNorm

# --------------------------------- LeNet
def LeNet(ph_Xin):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    numRawInputChannels = GetNumRawInputChannels()

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x[1|3]. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, numRawInputChannels, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(ph_Xin, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    fc2_drop = tf.nn.dropout(fc2, g_tph.fc2_dropout_keep_rate)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, g_trainSets.n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(g_trainSets.n_classes))
    logits = tf.matmul(fc2_drop, fc3_W) + fc3_b

    return logits

#--------------------------------- TrainingPipeline
def DefineTFPlaceHolders():
    global g_tph
    numRawInputChannels = GetNumRawInputChannels()
    g_tph.XItems = tf.placeholder(tf.float32, (None, 32, 32, numRawInputChannels))
    g_tph.yLabels = tf.placeholder(tf.int32, (None))
    g_tph.one_hot_y = tf.one_hot(g_tph.yLabels, g_trainSets.n_classes)
    g_tph.fc2_dropout_keep_rate = tf.placeholder(tf.float32, name="fc2_dropout_keep_rate")

#--------------------------------- TrainingPipeline
def TrainingPipeline(dsTrainNorm, dsValidNorm):
    '''
    Training Pipeline
    '''
    global g_trainSets

    g_trainSets.logits = LeNet(g_tph.XItems)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=g_tph.one_hot_y, logits=g_trainSets.logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = g_TRAINRATE)
    training_operation = optimizer.minimize(loss_operation)

    #Model Evaluation
    correct_prediction = tf.equal(tf.argmax(g_trainSets.logits, 1), tf.argmax(g_tph.one_hot_y, 1))
    g_trainSets.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    g_trainSets.saver = tf.train.Saver()

    if g_doTrainModel:
        timerStart = time.time()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(dsTrainNorm.X)

            print("\nTraining...")

            for i in range(g_NUM_EPOCHS):
                dsTrainNorm.X, dsTrainNorm.y = sklearn.utils.shuffle(dsTrainNorm.X, dsTrainNorm.y)

                for offset in range(0, num_examples, g_BATCH_SIZE):
                    end = offset + g_BATCH_SIZE
                    batch_x, batch_y = dsTrainNorm.X[offset:end], dsTrainNorm.y[offset:end]

                    dictFeed = {
                        g_tph.XItems: batch_x,
                        g_tph.yLabels: batch_y,
                        g_tph.fc2_dropout_keep_rate: 0.5
                    }
                    sess.run(training_operation, feed_dict=dictFeed)

                validation_accuracy = evaluate(dsValidNorm.X, dsValidNorm.y, g_trainSets.accuracy_operation, g_tph.XItems, g_tph.yLabels)
                print("EPOCH {:02}... ".format(i + 1), end='', flush=True)
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))

            timerElapsedS = time.time() - timerStart
            print("Training took {:.2f} seconds".format(timerElapsedS))

            if g_doSaveModel:
                g_trainSets.saver.save(sess, g_sessionSavesName)
                print("Model saved")
            else:
                print("Model NOT saved...skipped.")

            return(sess)
    else:
        print("Training... Skipped...")

#--------------------------------- evaluate
def evaluate(X_data, y_data, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0

    sess = tf.get_default_session()

    for offset in range(0, num_examples, g_BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + g_BATCH_SIZE], y_data[offset:offset + g_BATCH_SIZE]

        dictFeed = {
            x: batch_x,
            y: batch_y,
            g_tph.fc2_dropout_keep_rate: 1.0
        }

        accuracy = sess.run(accuracy_operation, feed_dict=dictFeed)
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

#--------------------------------- EvalDataSet
def EvalDataSet(dsIn):
    with tf.Session() as sess:
        g_trainSets.saver.restore(sess, tf.train.latest_checkpoint(g_sessionSavesDir))
        test_accuracy = evaluate(dsIn.X, dsIn.y, g_trainSets.accuracy_operation, g_tph.XItems, g_tph.yLabels)
        return test_accuracy

#--------------------------------- EvalDataSets
def EvalDataSets(dsListIn):
    print("\nEval Model on datasets")

    for ds in dsListIn:
        test_accuracy = EvalDataSet(ds)
        print("Eval DataSet({}) Accuracy = {:.3f}".format(ds.name, test_accuracy))


#--------------------------------- EvalImageRecs
def EvalImageRecs(imageRecs, dictIDToLabel, dsTrainRaw):
    global g_trainSets
    print("\nEvalModelExtra: ", end='', flush=True)

    numTestImages = len(imageRecs)

    X_items =  np.array([imageRec.GetImage() for imageRec in imageRecs])
    X_items = GrayscaleNormalize(X_items)
    y_labels = [imageRec.labelId for imageRec in imageRecs]

    with tf.Session() as sess:
        g_trainSets.saver.restore(sess, tf.train.latest_checkpoint(g_sessionSavesDir))
        test_accuracy = evaluate(X_items, y_labels, g_trainSets.accuracy_operation, g_tph.XItems, g_tph.yLabels)
        print("Overall Test Accuracy = {:.3f}".format(test_accuracy))

    with tf.Session() as sess:
        g_trainSets.saver.restore(sess, tf.train.latest_checkpoint(g_sessionSavesDir))

        dictFeed = {
            g_tph.XItems: X_items,
            #y: batch_y,
            g_tph.fc2_dropout_keep_rate: 1.0,
        }

        sfmax = sess.run(tf.nn.softmax(g_trainSets.logits), feed_dict=dictFeed)
        topk = sess.run(tf.nn.top_k(tf.constant(sfmax), k=3))

        for imageIndex in range(numTestImages):
            imageRec = imageRecs[imageIndex]
            recStr = "Image[{}]=({:>10}) type {:02}: '{}'".format(imageIndex, imageRec.fileNameBase, imageRec.labelId, imageRec.labelStr)
            print(recStr)
            print("Top 3 Match IDs{} => probabilites: {}".format(topk[1][imageIndex], topk[0][imageIndex]))
            print()

    if (g_doSupressPlots):
        return

    fig, axes = plt.subplots(numTestImages, 4, figsize=g_figSize, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    matchRanksStr = ["1st Match", "2nd Match", "3rd Match"]

    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(dsTrainRaw.y, return_index=True)

    for imageIndex, imageRec in enumerate(imageRecs):
        title = "{}: {}".format(y_labels[imageIndex], dictIDToLabel[y_labels[imageIndex]])
        xLabel = "TestImg: {}".format(imageRec.fileNameBase)

        ax = axes[imageIndex][0]
        ax.imshow(imageRec.GetImage(), interpolation='sinc')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xLabel, fontsize=9)

        for matchRank, matchingLabelID in enumerate(topk.indices[imageIndex]):

            matchScore = topk.values[imageIndex][matchRank]
            matchingImageIndex = imageIndices[matchingLabelID]
            matchingImage = dsTrainRaw.X[matchingImageIndex]
            matchingLabelStr = dictIDToLabel[matchingLabelID]

            isMatchCorrect = (matchingLabelID == y_labels[imageIndex])
            matchColor = "green" if isMatchCorrect else "black"

            title = "{}: {}".format(matchingLabelID, matchingLabelStr)
            xLabel = "{}: {:0.3}".format(matchRanksStr[matchRank], matchScore)

            ax = axes[imageIndex][matchRank + 1]
            ax.imshow(matchingImage, interpolation='sinc')
            ax.set_title(title, fontsize=9, color=matchColor)
            ax.set_xlabel(xLabel, fontsize=9, color=matchColor)

    plt.show()

###################################### PLOTS ###########################
###################################### PLOTS ###########################
###################################### PLOTS ###########################
#--------------------------------- Plot_ImageRecs
def Plot_ImageRecs(imageRecs):
    if g_doSupressPlots:
        return

    figsize = (12,4)
    plotNumCols=4
    plotNumRows= math.ceil(len(imageRecs)/plotNumCols)
    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})

    for (ax, imageRec) in zip(axes.flat, imageRecs):
        recStr = "{:<10} type {:02}: '{}'".format(imageRec.fileNameBase, imageRec.labelId, imageRec.labelStr)
        ax.set_title(recStr, fontsize=8)
        ax.imshow(imageRec.GetImage(), cmap='viridis', interpolation='sinc')

    plt.tight_layout()
    #plt.ion()  # turns on interactive mode
    plt.show()

#--------------------------------- Plot_LabeledSampleImages
def Plot_LabeledSampleImages(imgSamples, dictIDToLabel, dsTrainRaw):
    if g_doSupressPlots:
        return

    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(dsTrainRaw.y, return_index=True)

    figsize = (12,18)
    plotNumCols=4
    plotNumRows=11
    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})

    for (ax, imageIndex) in zip(axes.flat, imageIndices):
        labelID = dsTrainRaw.y[imageIndex]
        labelStr = dictIDToLabel[labelID]
        title = "{}:'{}'".format(labelID, labelStr, imageIndex)
        ax.set_title(title, fontsize=8)
        ax.imshow(imgSamples[imageIndex], cmap='viridis', interpolation='sinc')

    plt.tight_layout()
    #plt.ion()  # turns on interactive mode
    plt.show()

#--------------------------------- compute_normal_histograms
def Plot_NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw):
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

#====================== Main() =====================
def Main():
    dictIDToLabel = ReadLabelDict(signLabelsCSVFileIn)
    dsTrainRaw, dsValidRaw, dsTestRaw = ReadTrainingSets(training_file, validation_file, testing_file, dictIDToLabel)
    DefineTFPlaceHolders()

    dsTrainNorm, dsValidNorm, dsTestNorm = GrayscaleNormalizeRawDataSets([dsTrainRaw, dsValidRaw, dsTestRaw])
    #GrayscaleNormalizeAll()

    Plot_LabeledSampleImages(dsTrainRaw.X, dictIDToLabel, dsTrainRaw)
    #Plot_LabeledSampleImages(dsTrainNorm.X) # Doesnt make sense to display this
    Plot_NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw)

    TrainingPipeline(dsTrainNorm, dsValidNorm)# Only does training based on g_doTrainModel
    EvalDataSets([dsTrainNorm, dsValidNorm, dsTestNorm])
    #EvalModel()

    imageRecs = LoadTestImages(g_finalTestFilesDirIn, dictIDToLabel)
    Plot_ImageRecs(imageRecs)
    EvalImageRecs(imageRecs, dictIDToLabel, dsTrainRaw)

#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main()