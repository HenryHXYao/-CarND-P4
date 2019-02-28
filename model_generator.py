import os
import csv
import numpy as np
from scipy import ndimage

# import the training data
samples = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# split the training and validation set
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def get_image(source_path):
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename
	image = ndimage.imread(current_path)	
	return image	

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
				center_image = get_image(batch_sample[0])
				images.append(center_image)                    # append center_image
				images.append(np.fliplr(center_image)) 		   # append flipped center_image
				images.append(get_image(batch_sample[1]))      # append left_image
				images.append(get_image(batch_sample[2]))      # append right_image

				correction = 0.4
				measurement = float(batch_sample[3])
				measurements.append(measurement)
				measurements.append(-measurement)
				measurements.append(measurement + correction)	
				measurements.append(measurement - correction)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train) 

# compile and train the model using the generator function
batch_size = 32

train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)     

# define the model(NvidiaNet)
from keras.models import Sequential
from keras.layers import Lambda,Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5),strides=(2, 2), activation = 'relu'))
model.add(Conv2D(36, (5, 5),strides=(2, 2), activation = 'relu'))
model.add(Conv2D(48, (5, 5),strides=(2, 2), activation = 'relu'))
model.add(Conv2D(64, (3, 3),strides=(1, 1), activation = 'relu'))
model.add(Conv2D(64, (3, 3),strides=(1, 1), activation = 'relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# train the model
from math import ceil
from keras.callbacks import TensorBoard
# generate tensorboard for visualization
tensorboard = TensorBoard(log_dir='./logs') 
model.compile(loss='mse', optimizer = 'adam')
model.fit_generator(train_generator, steps_per_epoch= ceil(len(train_samples)/batch_size),
validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=10, verbose = 1, callbacks=[tensorboard])

# save the model
model.save('model_track_1plus2.h5')
    
   
