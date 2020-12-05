# **Traffic Sign Recognition** 

## Writeup

### (12/2) written by sungwook LE

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[image9]: ./LeNet_Custom(Own)_Architecture.jpg "My own Architecture"
[image10]: ./learn_monitor.png "Learning Accuracy and Loss"
[image11]: ./input_img.png "Input Image (the first one)"
[image12]: ./1st_conv.png "1st converlution Filter Images"
[image13]: ./2nd_conv.png "2nd converlution Filter Images"
[image14]: ./exploratory_visualization_dataset.png
[image15]: ./New_image_set.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

My Results: The Last EPOCHs Validation Accuaracy reaches 90%

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/SungwookLE/udacity_TrafficSign_Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43 (number of class)

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
![alt text][image14]
![alt text][image11]
![alt text][image12]
This is the Conv1 activation feature image

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The Total architecture that I made is like that
![alt text][image9]
It is similar to LeNet. Thus I call it 'def Lenet_Custom'

As a first step, I need the normalization input image
Thus, using (image-128)/128, all pixel data rearrange to mean(average) '0'

Second, First layer is convolution filter (conv1)
the filter size is (w,h,channel, depth) = (2,2,3,20), Padding = 'valid'.
Therefore, the output size is (16,16,20)

The resultant filter image looks as follow,
![alt text][image12]

Third, Max_pooling is designed for anti overfit
filter size is (ea, w, h, depth)=(1,3,3,1)
the output size is (14,14,20)

Fourth, convolution filter is put on the architecture,
the filter size is (7,7,20,20)
therfore, the output size is (8,8,20)

The resultant filter image looks as follow,
![alt text][image13]

Fifth is average pooling for anti overfit.
filter size is (1,3,3,1)
output size is (6,6,20)

Sixth, flatten the data matrix. 
That feature is input of fully connected neural network.

Seventh, Fully connected1 Layer
the weight size is (720,400)
and activation function is relu function
the last of this fc layer, drop out techniq is used to anti overfit

Eighth, Fully connected2 Layer
the weight size is (720,250)
and activation function is relu function
the last of this fc layer, drop out techniq is used to anti overfit

Nineth, Fully connected3 Layer
the weight size is (250,120)
and activation function is relu function
the last of this fc layer, drop out techniq is used to anti overfit

Finally, output layer
the size is (input_n, classes_n) = (120,43) 
output is 43 logits

and to pick the best candidate among output nodes,
the softmax function is used.

That's it


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 2x2x3x20 	| 2x2 stride, valid padding, outputs 16x16x20 	|
| RELU					|												|
| Max pooling			| 3x3 stride,  outputs 14x14x20 				|
| Convolution 7x7x20x20	| 1x1 stride, valid padding, outputs 8x8x20 	|
| RELU					|												|
| Avg pooling			| 3x3 stride,  outputs 6x6x20 					|
| Flatten 				| tf.contrib.layers.flatten(pool2)				|
| Fully connected1		| 720x400, with relu activation and dropout		|
| Fully connected2		| 400x250, with relu activation and dropout		|
| Fully connected3		| 250x120, with relu activation and dropout		|
| Output Layer			| 120x43, 43 is number of unique class label	|
| Softmax				| etc.        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an .... follow
Let's look the real code with me.

EPOCHS = 50 # EPOCHS means iteration, the bigger the number of EPOCHS , the better accuary (most cases)
BATCH_SIZE = 200 # the small group of total input data
rate = 0.001 # Learn_rate should be well picked, not high, not low
beta = 0.001 # it is for l2_normalization, the weight of loss_function

logits = LeNet_Custom(x, keep_prob) #This is Deep Learning Network
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y ) #Softmax is used to pick the best candidate

regularizer=0
for i in range(len(weights_bundle)):
    regularizer+= tf.nn.l2_loss(weights_bundle[i]) #This is L2_Regularization for anti overfit to improve the performance

loss_operation = tf.reduce_mean(cross_entropy+beta*regularizer) #You can see the beta*regularizer, beta mean weigh value 

optimizer = tf.train.AdamOptimizer(learning_rate = rate) # AdamOptimizer is used to GradientDescent
training_operation = optimizer.minimize(loss_operation) # Minimize the loss 

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 92%
* validation set accuracy of 89% 
* test set accuracy of 87%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
: To improve the total accuaracy, i tune the filter size(conv) and layer number etc.

* What were some problems with the initial architecture? 
: Training set accuray was good, but the test accuracy was too low,
Thus, I use dropout method, and L2 regularization,
Then, it is a liite improved

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training seut low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why? learn_rate is first tune parameter
after that, i tuned the keep_prob variable for dropout, and the filter size and layer numbers
to improve the accuaracy!

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
dropout is well worked i think. And I think the first information(size) must be well maintain, it the size become too small,
then the total performance will be low.

If a well known architecture was chosen:
* What architecture was chosen? my own architecture is based on LeNet
* Why did you believe it would be relevant to the traffic sign application? Yeap. CNN is good image learner
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

![alt text][image10]
the architecture (learning) is well worked as you can see


### Test a Model on New Images (Update Here 12/3)

** Feedback(12/3)
1)Test a Model on New Images
The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.
--> New Images are used, and get prediction value. 

2)The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.
--> well under the new_image_set, my architecture performance get worse (it is 60%)

3)The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.
--> look below the softmax results

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

This is the new_image set from web seaching
![alt text][image15]

The results is follow: (well work)

Prediction(that it predict):
35              Ahead only
0     Speed limit (20km/h)
34         Turn left ahead
40    Roundabout mandatory
1     Speed limit (30km/h)
Name: SignName, dtype: object

Label(that I want):
35               Ahead only
7     Speed limit (100km/h)
34          Turn left ahead
40     Roundabout mandatory
5      Speed limit (80km/h)
Name: SignName, dtype: object

See Prediction Value as Follow: (SoftMax results)
[Output Logits]-----------------------------------------------     [CLASS]--------
[ 36.46881866  21.60078239  18.98638916  17.70350838  11.37428761],[35 33 36 40 34]
[ 11.21093655   8.41795158   7.66519165   7.44350815   3.90690446],[ 0 28 29  1 38]
[ 22.94396019  18.64719582  13.03759384  11.4757843   10.07413483],[34 35 33 38 30]
[ 26.62835693  13.21689034   9.51245213   9.50981998   9.50655174],[40 41 32 42 37]
[ 11.07282066   9.47307968   7.37226915   6.87585402   6.34128428],[1 5 2 0 3]

--> Shuffle Accuracy: 60.0 % !!!



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


![alt text][image12]
![alt text][image13]
