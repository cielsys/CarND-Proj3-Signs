import os
import pickle
import csv
import matplotlib
import numpy as np
import math
import time
from collections import OrderedDict

import sklearn.utils
import tensorflow as tf
from tensorflow.contrib.layers import flatten

#%%time
#config = tf.ConfigProto()
#config.gpu_options.allocator_type = 'BFC'
#with tf.Session(config = config) as s:

#====================== GLOBALS =====================
# Lower the verbosity of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, or FATAL

# For testing only
g_doTests = False
g_doPlots = False
g_doUseDevDupeSet = True

g_doTrainModel = True
g_doSaveModel = True
g_doLoadModel = False

g_NUM_EPOCHS = 3
g_BATCH_SIZE = 64
g_TRAINRATE = 0.001
g_NUMOUTPUTS = 43

g_doConvertGray = True

g_TrainingFilesDirIn = "./Assets/training/"

signLabelsCSVFileIn = g_TrainingFilesDirIn + "signnames.csv"
training_file = g_TrainingFilesDirIn + "train.p"
validation_file = g_TrainingFilesDirIn + "valid.p"
testing_file = g_TrainingFilesDirIn + "test.p"

g_trainSets = type("DatasetContainer", (object,), {})

#----------------------- checkGPU()
def checkGPU():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

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
    global g_trainSets

    #DEBUG DUPE
    if g_doUseDevDupeSet:
        print("===> Using DevDupeSet!!! <=== \n")
        infile = validation_file
        with open(infile, mode='rb') as f:
            dupeSet = pickle.load(f)
        g_trainSets.X_train_raw, g_trainSets.y_train_raw = dupeSet['features'], dupeSet['labels']
        g_trainSets.X_test_raw, g_trainSets.y_test_raw = dupeSet['features'], dupeSet['labels']
        g_trainSets.X_valid_raw, g_trainSets.y_valid_raw = dupeSet['features'], dupeSet['labels']

    else:
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        g_trainSets.X_train_raw, g_trainSets.y_train_raw = train['features'], train['labels']
        g_trainSets.X_valid_raw, g_trainSets.y_valid_raw = valid['features'], valid['labels']
        g_trainSets.X_test_raw, g_trainSets.y_test_raw = test['features'], test['labels']


    # TODO: Number of training examples
    n_train = len(g_trainSets.X_train_raw)

    # TODO: Number of validation examples
    n_validation = len(g_trainSets.X_valid_raw)

    # TODO: Number of testing examples.
    n_test = len(g_trainSets.X_test_raw)

    # TODO: What's the shape of an traffic sign image?
    image_shape = g_trainSets.X_train_raw[0].shape

    # TODO: How many unique classes/labels there are in the dataset.
    g_trainSets.n_classes = len(set(g_trainSets.y_train_raw))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", g_trainSets.n_classes)

#--------------------------------- ReadLabelDict
def ReadLabelDict():
    '''
    Create dictionaries from the csv. Assumes line 0 header and unique indices in order
    '''
    global g_trainSets

    with open(signLabelsCSVFileIn, mode='r') as infile:
        infile.readline() # Skip header lin 0
        reader = csv.reader(infile)
        g_trainSets.dictIDToLabel = OrderedDict( (int(row[0]), row[1]) for row in csv.reader(infile) )
        g_trainSets.labelsStr = g_trainSets.dictIDToLabel.values()
        g_trainSets.labelsIDs = g_trainSets.dictIDToLabel.keys()
        #dictLabelToIndex = OrderedDict( (row[1], int(row[0])) for row in csv.reader(infile) )

#--------------------------------- GrayscaleNormalize
def ShuffleRawInPlace():
    global g_trainSets
    g_trainSets.X_train_raw, g_trainSets.y_train_raw = sklearn.utils.shuffle(g_trainSets.X_train_raw, g_trainSets.y_train_raw)
    g_trainSets.X_valid_raw, g_trainSets.y_valid_raw = sklearn.utils.shuffle(g_trainSets.X_valid_raw, g_trainSets.y_valid_raw)
    g_trainSets.X_test_raw, g_trainSets.y_test_raw = sklearn.utils.shuffle(g_trainSets.X_test_raw, g_trainSets.y_test_raw)

#--------------------------------- GrayscaleNormalize
def GrayscaleNormalize(dsIn):
    doConvert = False
    dsInf = tf.cast(dsIn, dtype=tf.float32)

    if g_doConvertGray:
        dsInf = tf.image.rgb_to_grayscale(dsInf)

    dsInf = tf.map_fn(lambda img: tf.image.per_image_standardization(img),  dsInf, dtype=tf.float32)

    dsOut = []
    with tf.Session() as sess:
        dsOut = dsInf.eval()
    return dsOut

# --------------------------------- GrayscaleNormalize
def GrayscaleNormalizeAll():
    global g_trainSets

    print ("\nNormalizing...", end='', flush=True)
    timerStart = time.time()

    if g_doUseDevDupeSet:
        g_trainSets.X_train_norm = GrayscaleNormalize(g_trainSets.X_train_raw)
        g_trainSets.X_valid_norm = g_trainSets.X_train_norm
        g_trainSets.X_test_norm = g_trainSets.X_train_norm

    else:
        g_trainSets.X_train_norm = GrayscaleNormalize(g_trainSets.X_train_raw)
        g_trainSets.X_valid_norm = GrayscaleNormalize(g_trainSets.X_valid_raw)
        g_trainSets.X_test_norm = GrayscaleNormalize(g_trainSets.X_test_raw)


    g_trainSets.y_train_norm = g_trainSets.y_train_raw
    g_trainSets.y_valid_norm = g_trainSets.y_valid_raw
    g_trainSets.y_test_norm = g_trainSets.y_test_raw

    timerElapsedS = time.time() - timerStart
    print ("{:.2f} seconds".format(timerElapsedS))

# --------------------------------- LeNet
def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

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

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, g_NUMOUTPUTS), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(g_NUMOUTPUTS))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


#--------------------------------- evaluate
def evaluate(X_data, y_data, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, g_BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + g_BATCH_SIZE], y_data[offset:offset + g_BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


#Features and labels
#x = tf.placeholder(tf.float32, (g_BATCH_SIZE, 32, 32, 1))
#y = tf.placeholder(tf.int32, (g_BATCH_SIZE))
#one_hot_y = tf.one_hot(y, g_NUMOUTPUTS)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, g_NUMOUTPUTS)

#--------------------------------- TrainingPipeline
def TrainingPipeline():
    '''
    #Training Pipeline
    '''
    #Features and labels
    #x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    #y = tf.placeholder(tf.int32, (None))
    #one_hot_y = tf.one_hot(y, 43)

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = g_TRAINRATE)
    training_operation = optimizer.minimize(loss_operation)

    #Model Evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    g_trainSets.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    g_trainSets.saver = tf.train.Saver()

    if g_doTrainModel:
        timerStart = time.time()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(g_trainSets.X_train_norm)

            print("Training...")

            for i in range(g_NUM_EPOCHS):
                g_trainSets.X_train_norm, g_trainSets.y_train_norm = sklearn.utils.shuffle(g_trainSets.X_train_norm, g_trainSets.y_train_norm)

                for offset in range(0, num_examples, g_BATCH_SIZE):
                    end = offset + g_BATCH_SIZE
                    batch_x, batch_y = g_trainSets.X_train_norm[offset:end], g_trainSets.y_train_norm[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

                validation_accuracy = evaluate(g_trainSets.X_valid_norm, g_trainSets.y_valid_norm, g_trainSets.accuracy_operation, x, y)
                print("EPOCH {:02}... ".format(i + 1), end='', flush=True)
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))

            timerElapsedS = time.time() - timerStart
            print("Training took {:.2f} seconds".format(timerElapsedS))

            if g_doSaveModel:
                g_trainSets.saver.save(sess, './lenet')
                print("Model saved")
            else:
                print("Model NOT saved...skipped.")

            return(sess)
    else:
        print("Training... Skipped...")

#--------------------------------- EvalModel
def EvalModel():
    print("\nEvalModel:")
    with tf.Session() as sess:
        g_trainSets.saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(g_trainSets.X_test_norm, g_trainSets.y_test_norm, g_trainSets.accuracy_operation, x, y)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    with tf.Session() as sess:
        g_trainSets.saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(g_trainSets.X_valid_norm, g_trainSets.y_valid_norm, g_trainSets.accuracy_operation, x, y)
        print("Validation Accuracy = {:.3f}".format(test_accuracy))

    with tf.Session() as sess:
        g_trainSets.saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(g_trainSets.X_train_norm, g_trainSets.y_train_norm, g_trainSets.accuracy_operation, x, y)
        print("Train Accuracy = {:.3f}".format(test_accuracy))

###################################### PLOTS ###########################
###################################### PLOTS ###########################
###################################### PLOTS ###########################

#--------------------------------- Plot_LabeledSampleImages
def Plot_LabeledSampleImages(imgSamples):
    global g_trainSets
    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(g_trainSets.y_train_raw, return_index=True)

    figsize = (12,18)
    plotNumCols=4
    plotNumRows=11
    fig, axes = plt.subplots(plotNumRows, plotNumCols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})

    for (ax, imageIndex) in zip(axes.flat, imageIndices):
        labelID = g_trainSets.y_train_raw[imageIndex]
        label = g_trainSets.dictIDToLabel[labelID]
        title = "{}:'{}'".format(labelID, label, imageIndex)
        ax.set_title(title, fontsize=8)
        ax.imshow(imgSamples[imageIndex], cmap='viridis', interpolation='sinc')

    plt.tight_layout()
    #plt.ion()  # turns on interactive mode
    plt.show()

#--------------------------------- compute_normal_histograms
def Plot_NumTrainingImagesHistogram():
    global g_trainSets
    # Get full set of ID's and the corresponding (first) image indices for each of of the IDs
    imageLabelIDs, imageIndices = np.unique(g_trainSets.y_train_raw, return_index=True)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Number of training images")
    #a = np.random.randint(2, size=1000)
    #b = np.random.randint(2, size=1000)
    bins=np.arange(g_trainSets.n_classes+1)-0.5
    plt.hist((g_trainSets.y_train_raw, g_trainSets.y_test_raw, g_trainSets.y_valid_raw), bins = bins, label = ("train", "test", "valid"))
    plt.legend()
    plt.xticks(imageLabelIDs)
    ax = plt.gca()
    histLables = ["{} : {}".format(label, labelID) for label, labelID in zip(g_trainSets.labelsStr, g_trainSets.labelsIDs)]
    ax.set_xticklabels(histLables, rotation=45, rotation_mode="anchor", ha="right")
    fig.subplots_adjust(bottom=0.5)
    #plt.ion()  # turns on interactive mode
    plt.show()

###################################### TESTS ###########################
###################################### TESTS ###########################
###################################### TESTS ###########################

#====================== Main() =====================
def Main():
    if g_doTests:
        checkGPU()

    ReadTrainingSets()
    ReadLabelDict()
    GrayscaleNormalizeAll()
    if g_doPlots:
        Plot_LabeledSampleImages(g_trainSets.X_train_raw)
        #Plot_LabeledSampleImages(g_trainSets.X_train_norm) # Doesnt make sense to display this
        Plot_NumTrainingImagesHistogram()

    TrainingPipeline()# Only does training based on g_doTrainModel


    EvalModel()

#====================== Main Invocation =====================
if ((__name__ == '__main__')):
    Main()