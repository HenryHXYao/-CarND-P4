# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/graph.png "Model Visualization"
[image2]: ./images/center.jpg "center_image"
[image3]: ./images/left.jpg "left Image"
[image4]: ./images/center.jpg "center Image"
[image5]: ./images/right.jpg "right Image"
[image6]: ./images/center.jpg "center Image"
[image7]: ./images/center_flip.jpg "center flip Image"
[image8]: ./images/train_valid_loss.png "loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Data Collection, Model Architecture and Training Strategy

#### Data Collection

Training data was chosen to keep the vehicle driving on the road. To capture good driving behavior, I recorded 4 laps on track-1 using center lane driving. Two of the 4 laps were clockwise and the others were counter-clockwise. There were 3442 center images in total and here is an example center image:

![alt text][image2]

When the center images were recorded while driving, the left and right images were also captured. So I used the left and right images to teach the vehicle to go back to the center line with correction = 0.1. 

The reason to choose left and right images rather than recording recovering driving from the sides of the road are:
* The left and right images were recorded all along the entire track, so it could provide more data than recording recovering driving at some spots on the track.
* You didn't need to teach the vehicle to go back to the center manually, which is quite exhausting compared to driving on the center line.

The left and right images provided 6884 samples, which is twice as the number of the center images. Here are the images captured by the left, center and right cameras:

|left|center|right|
|-|-|-|
|![alt text][image3]|![alt text][image4]|![alt text][image5]|

Finaly, I flipped the center images and angles to augment the data set. This approach provided additional 3442 images for training. Here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

So I had 13768 samples in total. I then preprocessed this data by randomly shuffling the data set and splitting 20% of the data into a validation set. 

---

#### Model Architecture

* **Solution Design Approach**

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

* **Final Model Architecture**
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray normalized image   							| 
| Convolution 5x5     	| 80 filters, 1x1 stride, valid padding, outputs 28x28x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x80				|
| Convolution 5x5	    | 80 filters, 1x1 stride, valid padding, outputs 10x10x80				|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x80				|
|Flatten |      outputs 2000           |
| Fully connected | ouptuts	120|
| RELU					|												|
| Dropout					|												|
| Fully connected | ouptuts	84	|
| RELU					|												|
| Dropout					|												|
| Fully connected | ouptuts	43	|
| Softmax				| 			|

Here is a visualization of the architecture from the tensorboard

![alt text][image1]


---

#### Training Strategy
Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image8]
