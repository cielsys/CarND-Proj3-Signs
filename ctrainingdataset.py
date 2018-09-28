import os
import glob
import pickle
import csv
import re
import time
from collections import OrderedDict
import copy

import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

g_ConvertImagesToTensors = False

#====================== CTrainingDataSet() =====================
class CTrainingDataSet():
    # ----------------------- ctor
    def __init__(self, name="NoName", pickleFileNameIn = None, dictIDToLabel = None):
        self.X = []
        self.y = []
        self.yStrings = []
        self.imageNames = []

        self.numLabels = 0
        self.testAccuracy = -1.0
        self.count = 0
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
            self.yStrings = [self.dictIDToLabel[yval] for yval in self.y]
            self.count = len(self.y)
            self.pickleFileNameIn = pickleFileNameIn
            imageNameBase = os.path.basename(pickleFileNameIn)
            self.imageNames = ["{}[{}]".format(imageNameBase, imageIndex) for imageIndex in range(self.count)]
            if (g_ConvertImagesToTensors):
                self.X = tf.image.convert_image_dtype(self.X, dtype=tf.float32)

    #--------------------------------- Concatenate
    def Concatenate(self, dsOther):
        # self.X.concatenate(dsOther.X)
        self.X = np.concatenate([self.X, dsOther.X])
        self.y = np.concatenate([self.y, dsOther.y])
        # self.yStrings.concatenate(dsOther.yStrings)
        # self.imageNames.concatenate(dsOther.imageNames)
        #self.X += dsOther.X
        #self.y  dsOther.y
        self.yStrings += dsOther.yStrings
        self.imageNames += dsOther.imageNames
        self.count = len(self.y)

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

    #--------------------------------- DebugTruncate
    def SegregateByLabel(self):
        print("\nSegregating {} images of dataset '{}' into {} datasets...".format(self.count, self.name, self.numLabels))
        timerStart = time.time()

        listDSSegregated = [CTrainingDataSet(name= "SegLabel{:02}".format(labelID), dictIDToLabel = self.dictIDToLabel) for labelID in self.labelsIDs]
        
        for index in range(self.count):
            cury = self.y[index]
            print("image({:05})->y({:02})".format(index,cury), end='\r', flush=True)
            segSetCur = listDSSegregated[cury]

            segSetCur.X.append(self.X[index])
            segSetCur.y.append(self.y[index])
            segSetCur.yStrings.append(self.yStrings[index])
            segSetCur.imageNames.append(self.imageNames[index])
            segSetCur.count +=1

        for (dsIndex, segSetCur) in enumerate(listDSSegregated):
            segSetCur.X = np.array(segSetCur.X)
            segSetCur.y = np.array(segSetCur.y)

        timerElapsedS = time.time() - timerStart
        print("\nDone in {:.1f} seconds".format(timerElapsedS))

        return listDSSegregated
    
#--------------------------------- CreateDataSetFromImageFiles
def CreateDataSetFromImageFiles(dirIn, dictIDToLabel, dataSetName="jpgFiles"):
    imageRecs = []
    fileExt = '*.jpg'
    fileSpec = dirIn + fileExt
    print("\nCreating dataset '{}' from files: {}... ".format(dataSetName, fileSpec), end='', flush=True)

    fileNamesIn = glob.glob(fileSpec)
    reExtractLabelNum = "(\d\d)\."

    dsOut = CTrainingDataSet(name=dataSetName, dictIDToLabel=dictIDToLabel)
    images = []

    for fileNameIn in fileNamesIn:
        labelId = labelIdStr = None
        reResult = re.search(reExtractLabelNum, fileNameIn)
        if reResult:
            labelId = int(reResult.groups()[0])
            labelStr = dictIDToLabel[labelId]

        fileNameBase = os.path.basename(fileNameIn)
        fileNameInFQ = os.path.abspath(fileNameIn)

        img =  mpimg.imread(fileNameInFQ)
        images.append(img)

        dsOut.y.append(labelId)
        dsOut.yStrings.append(labelStr)
        dsOut.imageNames.append(fileNameBase)

    dsOut.count = len(dsOut.y)
    dsOut.X = np.array(images)
    if (g_ConvertImagesToTensors):
        dsOut.X = tf.image.convert_image_dtype(dsOut.X, dtype=tf.float32)

    print("finished reading {} image files.".format(dsOut.count))
    return(dsOut)

#--------------------------------- ReadTrainingSets
def ReadTrainingSets(training_file, validation_file, testing_file, dictIDToLabel=None, truncatedTrainingSetSize=0):

    print("\nReading Training sets...")
    dsTrainRaw = CTrainingDataSet(name="TrainRaw", pickleFileNameIn = training_file, dictIDToLabel = dictIDToLabel)
    dsValidRaw = CTrainingDataSet(name="ValidRaw", pickleFileNameIn = validation_file, dictIDToLabel = dictIDToLabel)
    dsTestRaw  = CTrainingDataSet(name="TestRaw", pickleFileNameIn = testing_file, dictIDToLabel = dictIDToLabel)

    if truncatedTrainingSetSize > 0:
        print("*************** WARNING: Debug data sets truncated to size {}! *****************".format(truncatedTrainingSetSize))
        dsTrainRaw.DebugTruncate(truncatedTrainingSetSize)
        dsValidRaw.DebugTruncate(truncatedTrainingSetSize)
        dsTestRaw.DebugTruncate(truncatedTrainingSetSize)

    print("Raw training set size (train, validation, test) =({}, {}, {})".format(dsTrainRaw.count, dsValidRaw.count, dsTestRaw.count))
    print("Image data shape =", dsTrainRaw.GetXFeatureShape())
    print("Number of classes =", dsTrainRaw.GetDSNumLabels())

    if (g_ConvertImagesToTensors):
        dsTrainRaw.X = tf.image.convert_image_dtype(dsTrainRaw.X, dtype=tf.float32)
        dsValidRaw.X = tf.image.convert_image_dtype(dsValidRaw.X, dtype=tf.float32)
        dsTestRaw.X = tf.image.convert_image_dtype(dsTestRaw.X, dtype=tf.float32)

    return dsTrainRaw, dsValidRaw, dsTestRaw

#--------------------------------- ReadLabelDict
def ReadLabelDict(signLabelsCSVFileIn):
    '''
    Create dictionaries from the csv. Assumes line 0 header and unique indices in order
    '''

    with open(signLabelsCSVFileIn, mode='r') as infile:
        infile.readline() # Skip header line 0
        reader = csv.reader(infile)
        dictIDToLabel = OrderedDict( (int(row[0]), row[1]) for row in csv.reader(infile) )
    return dictIDToLabel

