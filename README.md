# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/RGB.PNG "Image components"
[image2]: ./examples/RGB_Gray.PNG "Added Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bhaubhau/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and csv library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I plotted a sample of 5 image sets and their RGB components, so as to get an idea of the different components of the training images
![alt text][image1]

I have also plotted the histogram on number of each type of images present per class as indicated in cell 6 of the notebook

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I added an extra grayscale layer to the images so that it makes identification of components like text displayed on the images as a whole 
![alt text][image2]

I also tried adding other layers hile Hue and saturation considering the fact that most of the signs have large portion of colors on them like Red, Blue etc, but observed those layers degrading the accuracy, so have not included these layers

As a last step, I normalized the image data so that the data has mean zero and equal variance


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model implementation with 4 layers input and added few other layers like dropout in it so as to improve the accuracy

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x4 RGB + Grayscaleimage   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input = 400. Output = 120        				|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84        				|
| RELU					|												|
| Dropout				|keep_prob=0.8									|
| Fully connected		| Input = 84. Output = 43        				|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with learning rate of 0.001 and Epochs=50, BatchSize=128. I also set an expected accuracy threshold, where if the model reached a desired threshold of 0.94, it saves the model
I tried training the model by modifying the hyper parameters like the number of Epochs. Saw that by increasing the number of epochs, the accuracy of the model reaches a specific point and then it starts oscillating. So once the model reached the specified threshold, I stopped the training and saved the model

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used implementation of existing LeNet model, which classifies nonMnist dataset of characters, and as the size of images available were close to those used by Lenet, this model was used. I tried using other techniques like introduction of dropout layers with different keep probabilities as well changing the output layer sizes, average pooling etc, and the model giving the desired accuracy was finaly chosen
My final model results were:
* validation set accuracy of 0.94
* test set accuracy of 0.92
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five german traffic signs I found on web are displayed in cell 14 of the file (https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html)
Some of the things to be considered in the images are 
1. The angle of signs like in case of the go straight or left, if the angle of sign gets mis aligned, the output results may not be correct
2. If the background is very large, the area of interest on the image may get destroyed when rescaling the image leading to wrong results
3. background objects like clouds, trees etc may cause inaccuracies

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction are displayed in cell 50 of the file (https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html)

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The bar chart of the five topmost probabilities is also displayed in cell 50 of the file (https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


