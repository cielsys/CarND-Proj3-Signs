# **Proj3: Neural Networks : Signs** 
### Submission writeup: ChrisL
Resubmission  2108-09-29 (Same day) Notes at EndOfFile
#### Reviewer Note:  

The main file for this project is [p3signs.py](./p3signs.py) <br/>
and [p3signsNB.html](p3signsNB.html) is the the best file for reviewing the runtime results, 
(but it is too large to view directly from github)<br/>
See **Usage and Implementation Notes** for complete file and folder descriptions, below.
---
<br/>

## **Overview**
Another fun but challenging assignment.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

The tensorFlow library is big and complicated and difficult to grapple with for beginners,
such as myself. It was slow going dealing with problems when then arose. But it was fun to 
persevere and have the NN succeed in the end.

---

### Data Set Summary & Exploration

#### 1. Data Sets basic info and stats
I used the training data sets from the udacity provided
 [traffic-signs-data.zip] [trainingsetdownload] file referenced in the [Project Instructions] [projectinstructions].
This data set is a subset from [German Trafﬁc Sign Recognition Benchmark] [GTSRB] that was
* Only 32x32x3 images
* Split into 3 sets: train, validation, test
* Each packaged into a python dictionary (whose full structure was never explained)
* And pypickle-serialized to the files train.p, valid.p, and test.p (in ```./Assets/training/```)<br/>

The serialized python dictionary also contained a ```.['labels']``` array of integers,
one per image, that encoded the **true** class, or label of the image. The integer
classifications serve as the 'y' values during training. However the downloaded dataset file
also contains a csv file [signnames.csv](./Assets/training/signnames.csv) that provides a corresponding
textual description string for each class. These strings are used in this document and in the application.
Each of the subsets had at least several examples of each class from the 43 classes
that are contained in the GTSRB.

Each image is a 32x32x3 RGB image. That is a 32x32 pixel image size  
with each pixel consisting 3 uint8 RGB values. Colorspace unkonwn.
The files encoded each image as a numpy array in the conventional way.
Thus it was easy to get basic stats on the datasets progromatically in python:

    Raw training set size (train, validation, test) =(34799, 4410, 12630)
    Image data shape = (32, 32, 3)
    Number of classes = 43

Each set of image arrays was contained, also, in a numpy array so that
the training set, for example, was a ndarray.shape = [34799, 32, 32, 3]
My code maintained the image datasets in the same structure except when
conversion was needed for use in tensorflow functions.

#### 2. Include an exploratory visualization of the dataset.
I wanted to get an idea of what the images were like and what the different classes/labels referred to 
so I generated a plot (with matplotlib/python) of the first image in the test set for each category.
As is evident some of the images are quite poor quality.

![alt text][trainsample]

These images have been 
resampled (sinc) up to larger than 32x32 and then again for display on the monitor
so is is difficult to see just how coarse is the resolution of the images. I zoomed
one of the images with no iterpolation so you can see the actual pixels that are
getting processed in this application. If you're near sighted like me
you can try taking you glasses on and off to get a DIY interpolation effect.


![Zoomed][02zoomed]


Also, I wanted to get an idea of the disribution of the images for each class/label, so I
generated this histogram. It is plain the the distribution of the training images is very uneven,
which is not ideal for CNN training. This will be remedied with image augmentation, below.

![alt text][trainhist]

#### 3. Dataset Augmentation Overview

To get an even distribution of training images I decided to perform image 
augmentation (see **Image Augmentation Specifics** below) on the underrepresented images of the training set.
The most represented class set was class 2 ('Speed limit 50k') with 2010 images.
I performed augmentation the images of each class creating new images until each set
had 2010 images. The images used for augmentation inputs were selected in order
from each set first to last in 'circular' order.


```
Segregating 34799 images of dataset 'TrainRaw' into 43 datasets...

Done in 0.1 seconds

Biggest image class is '2' with 2010 images.
label[00] has/needs = 180 / 1830 
label[01] has/needs = 1980 /  30 
label[02] has/needs = 2010 /   0 
label[03] has/needs = 1260 / 750 
label[04] has/needs = 1770 / 240 
label[05] has/needs = 1650 / 360 
label[06] has/needs = 360 / 1650 
label[07] has/needs = 1290 / 720 
label[08] has/needs = 1260 / 750 
label[09] has/needs = 1320 / 690 
label[10] has/needs = 1800 / 210 
label[11] has/needs = 1170 / 840 
label[12] has/needs = 1890 / 120 
label[13] has/needs = 1920 /  90 
label[14] has/needs = 690 / 1320 
label[15] has/needs = 540 / 1470 
label[16] has/needs = 360 / 1650 
label[17] has/needs = 990 / 1020 
label[18] has/needs = 1080 / 930 
label[19] has/needs = 180 / 1830 
label[20] has/needs = 300 / 1710 
label[21] has/needs = 270 / 1740 
label[22] has/needs = 330 / 1680 
label[23] has/needs = 450 / 1560 
label[24] has/needs = 240 / 1770 
label[25] has/needs = 1350 / 660 
label[26] has/needs = 540 / 1470 
label[27] has/needs = 210 / 1800 
label[28] has/needs = 480 / 1530 
label[29] has/needs = 240 / 1770 
label[30] has/needs = 390 / 1620 
label[31] has/needs = 690 / 1320 
label[32] has/needs = 210 / 1800 
label[33] has/needs = 599 / 1411 
label[34] has/needs = 360 / 1650 
label[35] has/needs = 1080 / 930 
label[36] has/needs = 330 / 1680 
label[37] has/needs = 180 / 1830 
label[38] has/needs = 1860 / 150 
label[39] has/needs = 270 / 1740 
label[40] has/needs = 300 / 1710 
label[41] has/needs = 210 / 1800 
label[42] has/needs = 210 / 1800 


Augmentation Done in 16.1 seconds
Combining datasets...Done. 51631 aug images added to 34799 ds TrainRaw for a total of 86430
```

After creating the 51631 augmented images they were added to original 34799 traing set images for a 
**complete equal-distibution training set of 86430 images.**

![alt text][trainhistaug]

#### 4. Image Augmentation Specifics
There were many options to choose from for random image augmentations: <br/>
Rotation, translation, perspective warp, resize, brignesst or contrast shifts, HSV shift, add noise
and no doubt many more.
Tensorflow has most these augmentations, some that are GPU efficient... In later
versions of tensor flow. 

As it turned out for my environement, performing these operations would require
either upgrading tensorflow, which I did not want to risk, or
converting datasets to and from other libraries such as openCF, PIL and
directly in numpy. After many efforts I settled on 2 relatively simple
augmentations, in order. 

* **Random Image rotation**<br/>
Using tf.contrib.image.rotate(img, rotAngleRad, interpolation='BILINEAR')
rotAngleRad was a uniform distribution +- 12 degrees.
Also, note that this transform does also an interpolation which itself
can be considered a kind of augmentation.

* **Random Brighness Shift**<br/>
Using``` tf.image.random_brightness(img, max_delta=0.1)```

* **Random Zoom**<br/>
After some forays I gave up on making this work, but I was close!

With more time I would have done the work to try out the various other augmentations.

Here is a sample of an original image and some augmented images
(Note: Img[1] is aug(Img[0], you can see brighness shift. Img[2,3] are augs of different originals. Img3 rotation is evident)

![augmentssmall][augmentssmall]<br/>
And here is a [larger sample][augmentsbig] of augments

---

### Design and Test a Model Architecture
I based my design on LeNet.
(borrowed image from the LeNet Lab)

![lenetlayers][lenetlayers]


#### 1. Image Preprocessing
All images fed into the tensorflow training or evaluation was normalized, for example (pixelNorm = pixel-128/128) as it
 is advised that feature values have average 0 value and unit standard deviation for fastest and most effective training. 
 Rather than use the crude formula above I found recomendationas for using
```tf.image.per_image_standardization(img)```
which places the average pixel value for entire image at 0.

Additionally at this step I permitted optional gray scale conversion first, as it is report that the is no advantage
to 3 channel RGB. I found that it made little difference.

#### 2. Model Design
I used a modified LeNet() model. The reason for this, honestly, is that we had a known working version of this
from the exercises. I then added 2 dropout layers at the last fully connected layers, and found that gave a good boost
to my results. When I next have time for this I will add some more dropout layers.

My final model consisted of the following layers:

 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   	(or 32x32x1 Grayscale) 	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten       		| Input = 5x5x64. Output = 400					|
| Fully connected		| Input = 400. Output = 120					 |
| RELU					|												|
| Dropout         		| 0.5 during training 							|
| Fully connected		| Input = 120 Output = 84   					|
| RELU					|												|
| Dropout         		| 0.5 during training 							|
| Fully Connected		| Input = 84. Output = 43

#### 3. Model Training

To train the model, I used AdamOptimizer that used mean cross entropy as the loss operation
to be minimized. Once again I used this optimizer soley because it was what worked
in the LeNet lab.
I used a variety of training rates and settled on a training rate of 
0.0009 in the submitted html. I never altered the defaults of
```
mu = 0
sigma = 0.1
```
Again, selection based on the fact that we had a working version from teaching module
labs
 

#### 4. Model Tweaking, finding a solution.

My approach was not disciplined. I tweaked mainly the main knobs of 
* number of epochs, 
* training rate and 
* batch size as well
as well as the 2 drop out rates. 

My initial values were to duplicate those in the LeNet lab solution.
During initial development I made gross adjustments to the values
as I worked on other parts of the application - I barely paid attention except to find
the deviant extremes that didn't work. I used this hapazard approach and by the time I had augmentation
and dropout implemented I was already achieving 0.96 validation values
with values near my final solution. 

Part of the reason for my extensive refactor was to create codebase with which
I could perform automated unattended test runs: 
Pick a set of interesting parameters, ie 'knobs to turn'
and select a range for each of them to test over then run the train/epoch eval with
those parameters and record all the results. 
Then look at the correlations
between the parameters and the validation results to see what is clearly bad combinations
or extremes. Then run the whole process again with the bad options eliminated and new finer
variations of combination selected.

Additionally comparing the [learning curve plots](https://www.dataquest.io/blog/learning-curves-machine-learning)
of all these runs could help to see which parameters combinations 
were yielding better accuracy conversion, again to help select new parameters for
another train/eval test... 

My dev computer is not too powerful, so training iterations were quite slow and 
once I found a working set that hit the target of 0.93 accuracy for the test set
I was, being overdue, not inclined to keep optimizing.
I also created different augmented train sets with different rotations and bright shifts.

My final model results were:

    Training: Epochs=22, TrainRate=0.0009, BatchSize=128
    Eval Model on datasets
    Eval DataSet(TrainCompleteNorm) Accuracy = 0.984
    Eval DataSet(ValidRawNorm) Accuracy = 0.967
    Eval DataSet(TestRawNorm) Accuracy = 0.945

---

### Extra Images
I retrieved 8 German traffic signs that I borrowed from other solutions:

![alt text][extraimages] 

#### 2. Extra image Eval Results
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Using one of my best trained models I got these softmax evaluations.
It was only a 50% correct rate.

```
Eval Model on datasets
Eval DataSet(jpgFilesNorm) Accuracy = 0.500

CalcSoftmaxTopK(jpgFilesNorm)
Image[0]=(    13.jpg) type 13: 'Yield'. 
     PASS! Top 3 Match IDs[13 12 14] => probabilites: [  1.00000000e+00   1.20530131e-11   1.52454687e-12]
Image[1]=(    03.jpg) type 03: 'Speed limit (60km/h)'. 
     FAIL. Top 3 Match IDs[40 17 37] => probabilites: [ 0.97900957  0.01432522  0.00362698]
Image[2]=(    14.jpg) type 14: 'Stop'. 
     PASS! Top 3 Match IDs[14 17 34] => probabilites: [  9.99999881e-01   1.47246283e-07   6.06446490e-11]
Image[3]=(    25.jpg) type 25: 'Road work'. 
     PASS! Top 3 Match IDs[25 30 20] => probabilites: [  9.99998331e-01   1.00913667e-06   3.62275529e-07]
Image[4]=(    17.jpg) type 17: 'No entry'. 
     PASS! Top 3 Match IDs[17 34 32] => probabilites: [  9.99994159e-01   4.82077985e-06   2.52489400e-07]
Image[5]=(    01.jpg) type 01: 'Speed limit (30km/h)'. 
     FAIL. Top 3 Match IDs[5 1 4] => probabilites: [ 0.63420701  0.35881561  0.00676099]
Image[6]=(    02.jpg) type 02: 'Speed limit (50km/h)'. 
     FAIL. Top 3 Match IDs[5 3 1] => probabilites: [ 0.57582909  0.39657083  0.01518974]
Image[7]=(    27.jpg) type 27: 'Pedestrians'. 
     FAIL. Top 3 Match IDs[18 27 26] => probabilites: [ 0.95076025  0.04278374  0.00645582]
```

#### 3. Extra image Eval Results Discussion.
The 50% results on the extra images are disappointing. Here are some thoughts on why.
1) **They aren't from test sets.**
Of the test/train/validate images I saw I was surpised how many images were nearly
identical to each other - the were frames from a video sequence. And I do not know how 
those sets were segregated but it seems possible that with our small subset and
the many duplications that we were overfitting to those 'kinds of images, and
the extra test images were not of 'those kinds'

2) The extra files a cleaner and brighter
I was surprised by the low quality of the pictures from the provided sets. 
There are many very dark images, very blurry images, very contrasty backgrounds and quite
a few with parts of other signs. The extra test set is much cleaner, focused, bright and uniform.
Perhaps it may be that the test set has been trained best for very poor quality issues.

One thing of note is that the speed limit signs are nearly identical for an image classifier
and my extra set contains 3 of them, which may account in part for the poor results.

![extrapredictions][extrapredictions]


### Usage and Implementation Notes
My project is non-typical in that it is not primarily a Notebook project.

#### Project file structure
**Python source Files**
The python code for this project is in 3 files
1. **p3signs.py:** The MAIN FILE <br/>
This main file contains all of the interesting neural net componements including
    * TrainingPipeline(), LeNet(), CalcSoftmaxTopK(), , NormalizeImageTensor(), evaluate()
    * CreateAugmentedDataSet(), AugmentImage(), ZoomImage()
    * Main()<br/>
    Which is the entry point when this file is run directly from python. 
    It is also an easy function to see the overall flow of the application.
 
2. **signplot.py** <br/>
Contains the matplotlib plotting routines for visualizing parts of this project.
3. **ctrainingdataset.py** <br/>
Contains a custom class primarily for keepin "X" features together with their correponsing
"y" labels. Also contains routines for serializing data to/from '.p' files

signsNB.py is export of the notebook for git history reasons and can be ignored.

**Jupyter Notebook.ipynb** The main notebook.<br/>
This notebook was created only for submission purposes. 
Because I find dev/debug difficult in a notebook and because many GPU problems seemed to arise
while using the notebook I developed outside the notebook environment. The code that is in this
notebook is just a duplication of the Main() routine from p3signs.py and some fixups
to coerce the application to run in the notebook. To see results from the NB see
[p3signsNB.html](p3signsNB.html)<br/>
signsNB.py is export of the notebook for git history reasons and can be ignored.

**Data files**
All i/o files are kept in the [./Assets/](./Assets/) directory. Here are the sub directories of interest.
* training/ Contains the training data sets as well as the augmented set **trainComplete.p**
* finalTest/ Contains the extra test images for the final step of this project testing.

#### Environment
This python 3.5 project was developed in pycharm community 2018.2 on Mint 18.3 with an Nvidia 960M
The program's shell/python environment was conda carnd-term1gpu
The NVidia and tensorflow specifics were too complicated to record

### Links
[MyRepo](https://github.com/cielsys/CarND-Proj3-Signs)<br/>
[OriginalRepo](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)<br/>
[Rubric](https://review.udacity.com/#!/rubrics/481/view) <br/>
[TrainingDataSet] [trainingsetdownload]<br/>
[Project Instructions] [projectinstructions]<br/>
[German Trafﬁc Sign Recognition Benchmark] [GTSRB]<br/>


### Resubmission Notes

1. The reviewer asked
```
Please add some words on what was the process that you ended up with the final model, e.g. 
what is the process of choosing batch size, number of layers? How do you choose the optimizer and why?
What should you do if the model is either overfitting or underfitting?
```
This puts me in a bind because my first submission answered this: 
*It was un disciplined trial and error.* But I will elaborate on how it happened and a more organzied
way to go about it.


2. Discuss the extra images
I did. But I will more. See
**3. Extra image Eval Results Discussion.** above.

[//]: # (WWW References)
[trainingsetdownload]: https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip
[projectinstructions]: https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/7ee8d0d4-561e-4101-8615-66e0ab8ea8c8/concepts/8cb6867c-f809-49b3-9bc1-afb409a112a7
[GTSRB]: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

[//]: # (Image References)

[//]: # (html resizeable image tag <img src='./Assets/examples/placeholder.png' width="480" alt="Combined Image" />)

[image1]: ./Assets/finalTest/01.jpg
[image2]: ./Assets/finalTest/02.jpg
[image3]: ./Assets/finalTest/03.jpg
[image4]: ./Assets/finalTest/04.jpg
[image5]: ./Assets/finalTest/13.jpg
[image6]: ./Assets/finalTest/14.jpg
[image7]: ./Assets/finalTest/17.jpg
[image8]: ./Assets/finalTest/25.jpg


[lenetlayers]:  ./Assets/writeupImages/lenetlayers.png

[trainhist]:  ./Assets/writeupImages/trainhist.png     
[trainhistaug]:  ./Assets/writeupImages/trainhistaug.png     
[extraimages]: ./Assets/writeupImages/extraimages.png     
[trainsample]:  ./Assets/writeupImages/trainsample.png     
[02zoomed]:  ./Assets/writeupImages/02zoomed.png
[augmentssmall]:  ./Assets/writeupImages/SampleAugmentsSmall.png
[augmentsbig]:  ./Assets/writeupImages/SampleAugmentsBig.png
[extrapredictions]:  ./Assets/writeupImages/predictions.png

