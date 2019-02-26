import csv
import numpy as np
from scipy import ndimage

# import the training data
lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

def get_image(source_path):
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename
	image = ndimage.imread(current_path)	
	return image	
      
images = []
measurements = []
for line in lines:
	center_image = get_image(line[0])
	images.append(center_image)            # append center_image
	images.append(np.fliplr(center_image)) # append flipped center_image
	images.append(get_image(line[1]))      # append left_image
	images.append(get_image(line[2]))      # append right_image

	correction = 0.1
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(-measurement)
	measurements.append(measurement + correction)	
	measurements.append(measurement - correction)	

X_train = np.array(images)
y_train = np.array(measurements)

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
from keras.callbacks import TensorBoard
# generate tensorboard for visualization
tensorboard = TensorBoard(log_dir='./logs') 
model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True,nb_epoch=10, callbacks=[tensorboard])

# save the model
model.save('model.h5')
    
   
