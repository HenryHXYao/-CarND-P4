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

#### 1. Data Collection

Training data was chosen to keep the vehicle driving on the road. To capture good driving behavior, I recorded 4 laps on track-1 using center lane driving. Two of the 4 laps were clockwise and the others were counter-clockwise. There were 3442 center images in total and here is an example center image:

![alt text][image2]

When the center images were recorded while driving, the left and right images were also captured at the same time. So I used the left and right images to teach the vehicle to go back to the center line with correction = 0.1. 

The reasons to choose left and right images rather than recording recovering driving from the sides of the road are:
* The left and right images were recorded all along the entire track, so it could provide more data than recording recovering driving at some spots on the track.
* You didn't need to teach the vehicle to go back to the center manually, which is quite exhausting compared to driving on the center line.
* In the simulator, it is possible to drive the car to the side of the road and then recover it to the center for the purpose of aquiring useful training data. However, it is almost impossible to do the same thing in the real world considering safety issues. Therefore, using the left and right images is an approach that has more potential to be implemented on real autonomous cars. 

The left and right images provided 6884 samples, which is twice as the number of the center images. Here are the images captured by the left, center and right cameras:

|left|center|right|
|-|-|-|
|![alt text][image3]|![alt text][image4]|![alt text][image5]|

Finaly, I flipped the center images and angles to augment the data set. This approach provided additional 3442 images for training. Here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

So I had 13768 samples in total. I then preprocessed this data by randomly shuffling the data set and splitting 20% of the data into a validation set. 

---

#### 2. Model Architecture

* **Solution Design Approach**

My first step was to use LeNet-5. After training process, I found that my first model had both high mean squared errors (about 0.05) on the training set and the validation set. This implied that the model was underfitting. When I ran my model on the simulator, the vehicle ran out of the road occasionally.

To combat the underfitting, I added more convolutional layers and fully connected layers to the model. Then the training error decreased to be 0.0042 and the validation error decreased to be 0.0058 which were both low, thus indicating good performance. Then I test my model again, this time the vehicle was able to drive autonomously around the track without leaving the road.

* **Final Model Architecture**

The final model architecture (model.py lines 44-57)(which refered to [Nvidia's CNN](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) ) consisted of a convolution neural network with the following layers and layer sizes, total parameters = 2712951:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   							| 
| Nomalization                   |      output  160x320x3                          |
| Cropping2D   |   output 65x320x3  |
| Convolution 5x5     	| 24 filters, 2x2 stride, valid padding, relu activation, outputs 31x158x24 	|		
| Convolution 5x5	    | 36 filters, 2x2 stride, valid padding, relu activation, outputs 14x77x36				|	
| Convolution 5x5	    | 48 filters, 2x2 stride, valid padding, relu activation, outputs 5x37x48				|	
| Convolution 3x3	    | 64 filters, 1x1 stride, valid padding, relu activation, outputs 3x35x64				|	
| Convolution 3x3	    | 64 filters, 1x1 stride, valid padding, relu activation, outputs 1x33x64				|	
|Flatten |      outputs 2112           |
| Fully connected | ouptuts	1164|
| Fully connected | ouptuts	100	|
| Fully connected | ouptuts	50	|
| Fully connected | ouptuts	10	|
| Fully connected | ouptuts	1	|

Here is a visualization of the architecture from the tensorboard:

![alt text][image1]

---

#### 3. Training Strategy
* The sample data was splitted into training set(80%) and validation set(20%). The model was trained on the training set and validated on the validation set (code line 64). The loss was selected to be the Mean Square Error. The following is the training loss and validation loss with 10 epochs:

![alt text][image8]

The training loss and validation loss were both low, therefore the model didn't have the problem of overfitting and underfitting.

* The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).
* The model was finally tested by running it through the simulator and the [results](https://github.com/HenryHXYao/CarND-P4/blob/master/video.mp4) showed that the vehicle could stay on the track. 

