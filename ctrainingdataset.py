import os
import glob
import pickle
import csv
import re
from collections import OrderedDict
import copy

import numpy as np
import matplotlib.image as mpimg

#====================== CTrainingDataSet() =====================
class CTrainingDataSet():
    # ----------------------- ctor
    def __init__(self, name="NoName", pickleFileNameIn = None, dictIDToLabel = None):
        self.X = None
        self.y = []
        self.testAccuracy = None
        self.yStrings = []
        self.numLabels = None
        self.count = None
        self.name = name
        self.imageNames = []
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
            self.count = len(self.y)
            self.pickleFileNameIn = pickleFileNameIn
            imageNameBase = os.path.basename(pickleFileNameIn)
            self.imageNames = ["{}[{}]".format(imageNameBase, imageIndex) for imageIndex in range(self.count)]

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

#--------------------------------- CreateDataSetFromImageFiles
def CreateDataSetFromImageFiles(dirIn, dictIDToLabel, dataSetName="jpgFiles"):
    imageRecs = []
    fileExt = '*.jpg'
    fileSpec = dirIn + fileExt
    print("\nCreating dataset '{}' from files: {}".format(dataSetName, fileSpec))

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

    return(dsOut)

#--------------------------------- ReadTrainingSets
def ReadTrainingSets(training_file, validation_file, testing_file, dictIDToLabel=None, truncatedTrainingSetSize=0):

    print("\nReading Training sets...")
    dsTrainRaw = CTrainingDataSet(name="TrainRaw", pickleFileNameIn = training_file, dictIDToLabel = dictIDToLabel)
    dsValidRaw = CTrainingDataSet(name="ValidRaw", pickleFileNameIn = validation_file, dictIDToLabel = dictIDToLabel)
    dsTestRaw  = CTrainingDataSet(name="TestRaw", pickleFileNameIn = testing_file, dictIDToLabel = dictIDToLabel)

    if truncatedTrainingSetSize > 0:
        print("*************** WARNING: Debug data sets truncated to size {}! *****************".format(g_truncatedTrainingSetSize))
        dsTrainRaw.DebugTruncate(truncatedTrainingSetSize)
        dsValidRaw.DebugTruncate(truncatedTrainingSetSize)
        dsTestRaw.DebugTruncate(truncatedTrainingSetSize)

    print("Raw training set size (train, validation, test) =({}, {}, {})".format(dsTrainRaw.count, dsValidRaw.count, dsTestRaw.count))
    print("Image data shape =", dsTrainRaw.GetXFeatureShape())
    print("Number of classes =", dsTrainRaw.GetDSNumLabels())

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

