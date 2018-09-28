import os
import glob
import pickle
import csv
import math
import time
import re
from collections import OrderedDict
import copy

import numpy as np
#import matplotlib
#import matplotlib.image
#import matplotlib.image as mpimg

import sklearn.utils
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import skimage
import skimage.transform

# Local Imports
from ctrainingdataset import CTrainingDataSet, CreateDataSetFromImageFiles, ReadTrainingSets, ReadLabelDict
import signplot

signplot.g_doShowPlots = False

#====================== GLOBALS =====================
seed = np.random.randint(42)

g_NUM_EPOCHS = 22
g_BATCH_SIZE = 128
g_TRAINRATE = 0.001

g_doConvertGray = True

# For debug testing only
g_truncatedTrainingSetSize = 0 # Default: 0 (Use full training sets). Truncation is for speedup of debug cycle
g_doTrainModel = True
g_doSaveModel = True

# I/O Directories
g_TrainingFilesDirIn = "./Assets/training/"
g_finalTestFilesDirIn = "./Assets/finalTest/"
g_sessionSavesDir = "./Assets/tfSessionSaves/"

# Input files
g_sessionSavesName = g_sessionSavesDir + "lenet"
g_signLabelsCSVFileIn = g_TrainingFilesDirIn + "signnames.csv"
g_training_file = g_TrainingFilesDirIn + "train.p"
g_validation_file = g_TrainingFilesDirIn + "valid.p"
g_testing_file = g_TrainingFilesDirIn + "test.p"

#====================== CODE =====================

# Lower the verbosity of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, or FATAL

#----------------------- checkGPU()
def checkGPU():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
#checkGPU()

#--------------------------------- GetNumRawInputChannels
def GetNumRawInputChannels():
    numRawInputChannels = 1 if g_doConvertGray else 3
    return numRawInputChannels

#--------------------------------- Normalize
def GreyScale(Xin):
    print ("\nConverting to greyscale...", end='', flush=True)
    Xout = tf.image.rgb_to_grayscale(Xin)
    print("done.")
    return(Xout)


#--------------------------------- Normalize
def NormalizeImageTensor(Xin):
    #XInfloat = tf.cast(Xin, dtype=tf.float32)

    if g_doConvertGray:
        Xin = tf.image.rgb_to_grayscale(Xin)

    XNorm = tf.map_fn(lambda img: tf.image.per_image_standardization(img),  Xin, dtype=tf.float32)

    # Convert back to ndarray
    Xout = []
    with tf.Session() as sess:
       Xout = XNorm.eval()

    return Xout

# --------------------------------- NormalizeDataSets
def NormalizeDataSets(listDSRaw):
    print ("\nNormalizing...convertGrayScale =={}. ".format(g_doConvertGray), end='', flush=True)
    timerStart = time.time()

    listDSNorm = []

    for dsRaw in listDSRaw:
        dsNorm = copy.copy(dsRaw)
        dsNorm.X = NormalizeImageTensor(dsRaw.X)
        dsNorm.name += "Norm"
        listDSNorm.append(dsNorm)

    timerElapsedS = time.time() - timerStart
    print ("{:.1f} seconds".format(timerElapsedS))

    return listDSNorm

#--------------------------------- TrainingPipeline
def DefineTFPlaceHolders(numLabels):
    tph = type("TensorFlowPlaceholders", (object,), {})

    numRawInputChannels = GetNumRawInputChannels()
    tph.XItems = tf.placeholder(tf.float32, (None, 32, 32, numRawInputChannels))
    tph.yLabels = tf.placeholder(tf.int32, (None))
    tph.one_hot_y = tf.one_hot(tph.yLabels, numLabels)
    tph.fc1_dropout_keep_rate = tf.placeholder(tf.float32, name="fc1_dropout_keep_rate")
    tph.fc2_dropout_keep_rate = tf.placeholder(tf.float32, name="fc2_dropout_keep_rate")

    return tph

# --------------------------------- LeNet
def LeNet(tph, numLabels):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    numRawInputChannels = GetNumRawInputChannels()

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x[1|3]. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, numRawInputChannels, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(tph.XItems, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

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

    fc1_drop = tf.nn.dropout(fc1, tph.fc1_dropout_keep_rate)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1_drop, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    fc2_drop = tf.nn.dropout(fc2, tph.fc2_dropout_keep_rate)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, numLabels), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(numLabels))
    logits = tf.matmul(fc2_drop, fc3_W) + fc3_b

    return logits

#--------------------------------- TrainingPipeline
def TrainingPipeline(dsTrainNorm, dsValidNorm, tfObjs, numEpochs, batchSize):
    '''
    Training Pipeline
    '''

    numLabels = dsTrainNorm.GetDSNumLabels()

    tfObjs.logits = LeNet(tfObjs.tph, numLabels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tfObjs.tph.one_hot_y, logits=tfObjs.logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = g_TRAINRATE)
    training_operation = optimizer.minimize(loss_operation)

    #Model Evaluation
    correct_prediction = tf.equal(tf.argmax(tfObjs.logits, 1), tf.argmax(tfObjs.tph.one_hot_y, 1))
    tfObjs.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tfObjs.saver = tf.train.Saver()

    if g_doTrainModel:
        timerStart = time.time()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = dsTrainNorm.count

            print("\nTraining for {} Epochs...".format(numEpochs))

            for i in range(numEpochs):
                dsShuffled_X, dsShuffled_y = sklearn.utils.shuffle(dsTrainNorm.X, dsTrainNorm.y)

                for offset in range(0, num_examples, batchSize):
                    end = offset + batchSize
                    batch_X, batch_y = dsShuffled_X[offset:end], dsShuffled_y[offset:end]

                    dictFeed = {
                        tfObjs.tph.XItems: batch_X,
                        tfObjs.tph.yLabels: batch_y,
                        tfObjs.tph.fc1_dropout_keep_rate: 0.4,
                        tfObjs.tph.fc2_dropout_keep_rate: 0.4
                    }
                    sess.run(training_operation, feed_dict=dictFeed)

                validation_accuracy = evaluate(dsValidNorm.X, dsValidNorm.y, tfObjs, batchSize)
                print("Epoch {:02}... ".format(i + 1), end='', flush=True)
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))

            timerElapsedS = time.time() - timerStart
            print("Training took {:.2f} seconds".format(timerElapsedS))

            if g_doSaveModel:
                tfObjs.saver.save(sess, g_sessionSavesName)
                print("Model saved")
            else:
                print("Model NOT saved...skipped.")

            return(sess)
    else:
        print("Training... Skipped...")

#--------------------------------- evaluate
def evaluate(X_data, y_data, tfObjs, batchSize):
    num_examples = len(y_data)
    total_accuracy = 0

    sess = tf.get_default_session()

    for offset in range(0, num_examples, batchSize):
        batch_x, batch_y = X_data[offset:offset + batchSize], y_data[offset:offset + batchSize]

        dictFeed = {
            tfObjs.tph.XItems: batch_x,
            tfObjs.tph.yLabels: batch_y,
            tfObjs.tph.fc1_dropout_keep_rate: 1.0,
            tfObjs.tph.fc2_dropout_keep_rate: 1.0,
        }

        accuracy = sess.run(tfObjs.accuracy_operation, feed_dict=dictFeed)
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

#--------------------------------- EvalDataSet
def EvalDataSet(dsIn, tfObjs, batchSize):
    with tf.Session() as sess:
        tfObjs.saver.restore(sess, tf.train.latest_checkpoint(g_sessionSavesDir))
        test_accuracy = evaluate(dsIn.X, dsIn.y, tfObjs, batchSize)
        dsIn.testAccuracy = test_accuracy
        return test_accuracy

#--------------------------------- EvalDataSets
def EvalDataSets(dsNormListIn, tfObjs, batchSize):
    print("\nEval Model on datasets")

    accuracyList = []
    for dsNormIn in dsNormListIn:
        test_accuracy = EvalDataSet(dsNormIn, tfObjs, batchSize)
        print("Eval DataSet({}) Accuracy = {:.3f}".format(dsNormIn.name, test_accuracy))
        accuracyList.append(test_accuracy)

    return(accuracyList)

#--------------------------------- CalcSoftmaxTopK
def CalcSoftmaxTopK(dsInNorm, tfObjs):
    print("\nCalcSoftmaxTopK({})".format(dsInNorm.name), end='', flush=True)

    with tf.Session() as sess:
        tfObjs.saver.restore(sess, tf.train.latest_checkpoint(g_sessionSavesDir))

        dictFeed = {
            tfObjs.tph.XItems: dsInNorm.X,
            #y: batch_y,
            tfObjs.tph.fc1_dropout_keep_rate: 1.0,
            tfObjs.tph.fc2_dropout_keep_rate: 1.0,
        }

        sfmax = sess.run(tf.nn.softmax(tfObjs.logits), feed_dict=dictFeed)
        topk = sess.run(tf.nn.top_k(tf.constant(sfmax), k=3))

        for imageIndex in range(dsInNorm.count):
            recStr = "Image[{}]=({:>10}) type {:02}: '{}'".format(imageIndex, dsInNorm.imageNames[imageIndex], dsInNorm.y[imageIndex], dsInNorm.yStrings[imageIndex])
            print(recStr)
            print("Top 3 Match IDs{} => probabilites: {}".format(topk[1][imageIndex], topk[0][imageIndex]))
            print()

    return topk

#--------------------------------- ScaleImage
def ZoomImage(imgIn, scaleMaxPercent = 10):
    scaleMaxPercent = scaleMaxPercent / 100.0
    (height, width, channels) = imgIn.shape

    scale = np.random.uniform(low=1.0 - scaleMaxPercent, high=1.0 + scaleMaxPercent, size=None)
    imgScaled = skimage.transform.rescale(imgIn, scale=scale, mode='constant', multichannel=True, anti_aliasing=True)

    #imgResized = skimage.transform.resize(imgScaled, (height, width), mode='constant', anti_aliasing=True)

    imgScaledF = tf.image.convert_image_dtype(imgScaled, dtype=tf.float32)
    ts = tf.image.resize_image_with_crop_or_pad(imgScaledF, height, width)
    with tf.Session() as sess:
        imgZoomed = ts.eval()

    return imgZoomed

#--------------------------------- Rotate
def AugmentImage(Xin, rotAngleMaxDeg=10):
    #print ("Rotating random({}) degrees...".format(rotAngleDegMax), end='', flush=True)
    #timerStart = time.time()
    #Xinfloat = tf.cast(np.array(Xin), dtype=tf.float32)

    (numImages, height, width, channels) = Xin.shape

    rotAngleMaxDegRad = rotAngleMaxDeg * math.pi / 180
    tRandomRot = tf.random_uniform([numImages], minval=-rotAngleMaxDegRad, maxval=rotAngleMaxDegRad, dtype=tf.float32)

    #XZoomed = np.array([ZoomImage(img, scaleMaxPercent=50) for img in Xin])

    #################### RANDOM ROTATION
    XRot = tf.contrib.image.rotate(Xin, tRandomRot, interpolation='BILINEAR')
    #XRot = tf.map_fn(lambda img: tf.contrib.image.rotate(img, rotAngleRad, interpolation='BILINEAR'),  Xin, dtype=tf.float32)


    #################### RANDOM BRIGHTNESS SHIFT
    XBright = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=0.3),  XRot, dtype=tf.uint8)
    #XBright = tf.image.random_brightness(XRot, tRandomBright)

    with tf.Session() as sess:
       Xout = XBright.eval()

    #timerElapsedS = time.time() - timerStart
    #print ("Rotate done. {:.1f} seconds".format(timerElapsedS))
    return Xout


# --------------------------------- CreateAugmentedDataSet
def CreateAugmentedDataSet(dsIn, numAugsNeeded):
    print("Augmenting data subset '{}'. ".format(dsIn.name), end="", flush=True)
    timerStart = time.time()

    dsAugOut = copy.copy(dsIn)
    dsAugOut.name += dsIn.name + ".Aug"
    #bigXinput = np.empty(shape=(0, 32, 32, 3))
    bigXinput = np.empty(shape=(numAugsNeeded, 32, 32, 3), dtype=np.uint8)
    dsAugOut.y = np.empty(shape=(numAugsNeeded))
    dsAugOut.yStrings = []

    tmpX = None
    startIndex = 0
    while numAugsNeeded > 0:
        numAugsCur = numAugsNeeded if numAugsNeeded < dsIn.count else dsIn.count
        endIndex = startIndex + numAugsCur

        bigXinput[startIndex:endIndex, :, :, :] = dsIn.X[ : numAugsCur, :, :, :]
        dsAugOut.y[startIndex:endIndex] = dsIn.y[ : numAugsCur]
        dsAugOut.yStrings += dsIn.yStrings[ : numAugsCur]

        startIndex += numAugsCur
        numAugsNeeded -= numAugsCur

        #tmpX = bigXinput

    dsAugOut.X = AugmentImage(bigXinput, rotAngleMaxDeg=10)
    dsAugOut.count = len(dsAugOut.X)

    timerElapsedS = time.time() - timerStart
    print("Augmentation data subset done in {:.1f} seconds".format(timerElapsedS))
    return dsAugOut

# --------------------------------- CreateAugmentedDataSets
def CreateAugmentedDataSets(dsIn):
    print("\nAugmenting dataset '{}'.".format(dsIn.name))
    timerStart = time.time()

    listDSInSegregated = dsIn.SegregateByLabel()
    (longestDSIndex, longestDS) = max(enumerate(listDSInSegregated), key = lambda tupIndDS: tupIndDS[1].count)

    print("\nBiggest image class is '{}' with {} images.".format(longestDSIndex, longestDS.count))
    targetLen = longestDS.count


    listDSAugmentsSegregated = []

    for yIndexCur, dsInSegCur in enumerate(listDSInSegregated):
        numAugsNeeded = targetLen - dsInSegCur.count
        print("label[{:02}] has/needs = {:3} / {:3}... ".format(yIndexCur, dsInSegCur.count, numAugsNeeded), end="", flush=True)
        dsAugCur = CreateAugmentedDataSet(dsInSegCur, numAugsNeeded)
        listDSAugmentsSegregated.append(dsAugCur)



    # Unsegregate augments, then combine with originals to make complete training set
    dsOutAugments = copy.copy(dsIn)
    dsOutAugments.name = "Aug({})".format(dsIn.name)
    dsOutAugments.X = []
    dsOutAugments.y = []
    dsOutAugments.yStrings = []
    dsOutAugments.imageNames = []

    for dsInSegCur in listDSAugmentsSegregated:
        dsOutAugments.X += dsInSegCur.X.tolist()
        dsOutAugments.y += dsInSegCur.y.tolist()
        dsOutAugments.yStrings += dsInSegCur.yStrings
        dsOutAugments.imageNames += dsInSegCur.imageNames
    dsOutAugments.X = np.array(dsOutAugments.X, dtype=np.uint8)
    dsOutAugments.y = np.array(dsOutAugments.y, dtype=np.uint8)
    dsOutAugments.count = len(dsOutAugments.y)

    dsComplete = copy.copy(dsIn)
    dsComplete.Concatenate(dsOutAugments)
    dsComplete.name += dsIn.name + "+Aug"

    timerElapsedS = time.time() - timerStart
    print("\nAugmentation Done in {:.1f} seconds".format(timerElapsedS))

    return listDSInSegregated, listDSAugmentsSegregated, dsOutAugments, dsComplete


#====================== Main() =====================
def Main():
    numEpochs = g_NUM_EPOCHS
    batchSize = g_BATCH_SIZE
    # Sort-of global object containers
    tfObjs = type("TensorFlowObjectsContainer", (object,), {})
    tfObjs.tph = None  # Assigned in DefineTFPlaceHolders()

    dictIDToLabel = ReadLabelDict(g_signLabelsCSVFileIn)

    dsTrainRaw, dsValidRaw, dsTestRaw = ReadTrainingSets(g_training_file, g_validation_file, g_testing_file, dictIDToLabel, truncatedTrainingSetSize = g_truncatedTrainingSetSize)

    listDSInSegregated, listDSOutAugmentedSegregated, dsOutAugments, dsTrainComplete = CreateAugmentedDataSets(dsTrainRaw)
    signplot.PlotListOfDataSets(listDSOutAugmentedSegregated, "augments")# **************DEBUG****************


    signplot.LabeledSampleImages(dsTrainRaw)
    signplot.NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw)

    #dsTrainNorm, dsValidNorm, dsTestNorm = NormalizeDataSets([dsTrainRaw, dsValidRaw, dsTestRaw])
    dsTrainNormComplete, dsValidNorm, dsTestNorm = NormalizeDataSets([dsTrainComplete, dsValidRaw, dsTestRaw])

    tfObjs.tph = DefineTFPlaceHolders(dsTrainRaw.GetDSNumLabels())
    TrainingPipeline(dsTrainNormComplete, dsValidNorm, tfObjs, numEpochs, batchSize)

    #TrainingPipeline(dsTrainNorm, dsValidNorm, tfObjs, numEpochs, batchSize)
    EvalDataSets([dsTrainNormComplete, dsValidNorm, dsTestNorm], tfObjs, batchSize)

    # Extra dataset
    dsExtraRaw = CreateDataSetFromImageFiles(g_finalTestFilesDirIn, dictIDToLabel)
    signplot.LabeledSampleImages(dsExtraRaw)

    dsExtraNorm, = NormalizeDataSets([dsExtraRaw])
    EvalDataSets([dsExtraNorm], tfObjs, batchSize)
    topk = CalcSoftmaxTopK(dsExtraNorm, tfObjs)
    signplot.PredictionComparison(dsExtraRaw, dsTrainRaw, topk)

#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main()