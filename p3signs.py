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
import matplotlib
import matplotlib.image
import matplotlib.image as mpimg

import sklearn.utils
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Local Imports
from ctrainingdataset import CTrainingDataSet, CreateDataSetFromImageFiles, ReadTrainingSets, ReadLabelDict
import signplot

signplot.g_doSupressPlots = False

#====================== GLOBALS =====================
g_NUM_EPOCHS = 15
g_BATCH_SIZE = 32
g_TRAINRATE = 0.001

g_doConvertGray = False

# For debug testing only
g_truncatedTrainingSetSize = 0 # Default: 0 (Use full training sets). Truncation is for speedup of debug cycle
g_doTrainModel = True
g_doSaveModel = True

# I/O Directories
g_TrainingFilesDirIn = "./Assets/training/"
g_finalTestFilesDirIn = "./Assets/finalTest/"
g_sessionSavesDir = "./Assets/tfSessionSaves/"
g_sessionSavesName = g_sessionSavesDir + "lenet"

# Input files
signLabelsCSVFileIn = g_TrainingFilesDirIn + "signnames.csv"
training_file = g_TrainingFilesDirIn + "train.p"
validation_file = g_TrainingFilesDirIn + "valid.p"
testing_file = g_TrainingFilesDirIn + "test.p"

# Global object containers
g_tfObjs = type("TensorFlowObjectsContainer", (object,), {})
g_tfObjs.tph = None # Assigned in DefineTFPlaceHolders()

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

#--------------------------------- TrainingPipeline
def DefineTFPlaceHolders(numLabels):
    tph = type("TensorFlowPlaceholders", (object,), {})

    numRawInputChannels = GetNumRawInputChannels()
    tph.XItems = tf.placeholder(tf.float32, (None, 32, 32, numRawInputChannels))
    tph.yLabels = tf.placeholder(tf.int32, (None))
    tph.one_hot_y = tf.one_hot(tph.yLabels, numLabels)
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

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

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
                        tfObjs.tph.fc2_dropout_keep_rate: 0.5
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
            tfObjs.tph.fc2_dropout_keep_rate: 1.0
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

#====================== Main() =====================
def Main():
    global g_tfObjs

    numEpochs = g_NUM_EPOCHS
    batchSize = g_BATCH_SIZE

    dictIDToLabel = ReadLabelDict(signLabelsCSVFileIn)
    dsTrainRaw, dsValidRaw, dsTestRaw = ReadTrainingSets(training_file, validation_file, testing_file, dictIDToLabel, truncatedTrainingSetSize = g_truncatedTrainingSetSize)

    g_tfObjs.tph = DefineTFPlaceHolders(dsTrainRaw.GetDSNumLabels())
    dsTrainNorm, dsValidNorm, dsTestNorm = GrayscaleNormalizeRawDataSets([dsTrainRaw, dsValidRaw, dsTestRaw])

    signplot.LabeledSampleImages(dsTrainRaw)
    signplot.NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw)

    TrainingPipeline(dsTrainNorm, dsValidNorm, g_tfObjs, numEpochs, batchSize)
    EvalDataSets([dsTrainNorm, dsValidNorm, dsTestNorm], g_tfObjs, batchSize)

    dsExtraRaw = CreateDataSetFromImageFiles(g_finalTestFilesDirIn, dictIDToLabel)
    signplot.LabeledSampleImages(dsExtraRaw)

    dsExtraNorm, = GrayscaleNormalizeRawDataSets([dsExtraRaw])
    EvalDataSets([dsExtraNorm], g_tfObjs, batchSize)

    topk = CalcSoftmaxTopK(dsExtraNorm, g_tfObjs)
    signplot.PredictionComparison(dsExtraRaw, dsTrainRaw, topk)

#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main()