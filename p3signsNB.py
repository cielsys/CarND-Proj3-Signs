
# coding: utf-8

# In[3]:


# Local Imports
#from ctrainingdataset import CTrainingDataSet, CreateDataSetFromImageFiles, ReadTrainingSets, ReadLabelDict
import signplot
import p3signs

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')
#%aimport ctrainingdataset
get_ipython().magic('aimport signplot')
get_ipython().magic('aimport p3signs')


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>\nfrom IPython.core.display import display, HTML\njnk = display(HTML("<style>.container { width:100% !important; }</style>"))')


# In[7]:


#====================== GLOBALS =====================
g_Args = p3signs.GetgArgs()
g_Args.numEpochs = 22
g_Args.batchSize = 128
g_Args.trainRate = 0.001
g_Args.doConvertGray = True

signplot.g_doShowPlots = True


# In[5]:


get_ipython().magic('autoreload 1')
#====================== Main() =====================
def Main(gArgs):
    
    #################### Sort-of global object containers
    tfObjs = type("TensorFlowObjectsContainer", (object,), {})
    tfObjs.tph = None  # Assigned in DefineTFPlaceHolders()

    
    #################### READ DATASETS
    dictIDToLabel = p3signs.ReadLabelDict(gArgs.signLabelsCSVFileIn)
    dsTrainRaw, dsValidRaw, dsTestRaw = p3signs.ReadTrainingSets(gArgs.trainingFileIn, gArgs.validationFileIn, gArgs.testingFileIn, dictIDToLabel, truncatedTrainingSetSize = gArgs.truncatedTrainingSetSize)
    signplot.LabeledSampleImages(dsTrainRaw)
    signplot.NumTrainingImagesHistogram(dsTrainRaw, dsValidRaw, dsTestRaw)
    

    
    #################### CREATE AUGMENTS
    if (gArgs.doComputeAugments):
        listDSInSegregated, listDSOutAugmentedSegregated, dsOutAugments, dsTrainComplete = p3signs.CreateAugmentedDataSets(dsTrainRaw)
        signplot.PlotListOfDataSets(listDSOutAugmentedSegregated, "augments")
        
        if(gArgs.doSaveTrainCompleteFile):
            print("Saving augment complete file {}".format(gArgs.trainingCompleteFile))
            dsTrainComplete.WritePickleFile(gArgs.trainingCompleteFile)
    else:
        print("Loading augment complete file {}".format(gArgs.trainingCompleteFile))
        dsTrainComplete = p3signs.CTrainingDataSet(name="TrainComplete", pickleFileNameIn = gArgs.trainingCompleteFile, dictIDToLabel = dictIDToLabel)

    signplot.NumTrainingImagesHistogram(dsTrainComplete, dsValidRaw, dsTestRaw)

    
    
    #################### NORM, TRAINING & EVAL
    dsTrainNormComplete, dsValidNorm, dsTestNorm = p3signs.NormalizeDataSets([dsTrainComplete, dsValidRaw, dsTestRaw], gArgs.doConvertGray)
    tfObjs.tph = p3signs.DefineTFPlaceHolders(dsTrainRaw.GetDSNumLabels(), gArgs)
    p3signs.TrainingPipeline(dsTrainNormComplete, dsValidNorm, tfObjs, gArgs)
    p3signs.EvalDataSets([dsTrainNormComplete, dsValidNorm, dsTestNorm], tfObjs, gArgs)
    
    

    #################### EXTRA DATASET
    dsExtraRaw = p3signs.CreateDataSetFromImageFiles(gArgs.finalTestFilesDirIn, dictIDToLabel)
    signplot.LabeledSampleImages(dsExtraRaw)

    dsExtraNorm, = p3signs.NormalizeDataSets([dsExtraRaw], gArgs.doConvertGray)
    p3signs.EvalDataSets([dsExtraNorm], tfObjs, gArgs)
    topk = p3signs.CalcSoftmaxTopK(dsExtraNorm, tfObjs, gArgs)
    signplot.PredictionComparison(dsExtraRaw, dsTrainRaw, topk)



# In[8]:


get_ipython().magic('autoreload 1')

#====================== Main Invocation =====================
Main(g_Args)

