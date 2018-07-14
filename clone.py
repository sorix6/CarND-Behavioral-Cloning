import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras.backend import tf as ktf
from keras.models import Model

csv_file = './new_recordings/driving_log.csv';

def imageLoading():
	lines = []

	with open(csv_file) as csvfile:
		reader = csv.reader(csvfile)
		
		for line in reader:
			lines.append(line)

	images = []
	measurements = []
	correction = 0.1 # this is a parameter to tune
	
	for line in lines:
		image_center = cv2.imread(line[0])
		steering_center = float(line[3])
		images.append(image_center)
		measurements.append(steering_center)
		
		image_left = cv2.imread(line[1])
		images.append(image_left)
		measurements.append(steering_center + correction)
		
		image_right = cv2.imread(line[1])
		images.append(image_right)
		measurements.append(steering_center - correction)
		
	return images, measurements

	
def augmentData(images, measurements):
	augmented_images, augmented_measurements = [], []
	
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement);
		augmented_images.append(cv2.flip(image, 1))
		augmented_measurements.append(measurement * -1.0)
		
	return augmented_images, augmented_measurements

	
## LeNet model
def leNet():
	model = Sequential()
	model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Lambda(converter))
	
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	# Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
	model.add(Convolution2D(6,5,5,activation="relu"))

	# Pooling. Input = 28x28x6. Output = 14x14x6.
	model.add(MaxPooling2D())

	# Layer 2: Convolutional. Output = 10x10x16.
	model.add(Convolution2D(6,5,5, activation="relu"))

	# Pooling. Input = 10x10x16. Output = 5x5x16.
	model.add(MaxPooling2D())

	# Flatten. Input = 5x5x16. Output = 400.
	model.add(Flatten())

	# Layer 3: Fully Connected. Input = 400. Output = 120.
	model.add(Dense(120))

	# Layer 4: Fully Connected. Input = 120. Output = 84.
	model.add(Dense(84))

	# Layer 5: Fully Connected. Input = 84. Output = 1.
	model.add(Dense(1))
    
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

	model.save('model_leNet.h5')
	exit()

## NVIDIA Model	
def nvidia():
	model = Sequential()
	model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
	
	model.add(Lambda(converter))
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	
	# Layer 1: Convolutional
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
	
	# Layer 2: Convolutional
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
	
	# Layer 3: Convolutional
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))

	# Layer 4: Convolutional
	model.add(Convolution2D(64,3,3, activation="relu"))
	
	# Layer 5: Convolutional
	model.add(Convolution2D(64,3,3, activation="relu"))
	
	# Flatten
	model.add(Flatten())

	# Layer 6: Fully Connected
	model.add(Dense(100))

	# Layer 7: Fully Connected
	model.add(Dense(50))

	# Layer 8: Fully Connected
	model.add(Dense(10))
	
	# Layer 9: Fully Connected
	model.add(Dense(1))
    
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

	model.save('model_nvidia.h5')
	exit()
	#return model

## convert the image to grayscale	
def converter(x):
    #x has shape (batch, width, height, channels)
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])

images, measurements = imageLoading()
augmented_images, augmented_measurements = augmentData(images, measurements)
	
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#leNet();

nvidia()


################################################################
#model = nvidia();


#print(model.summary()) ## print the model summary

## save images containing intermediary outputs of treatment layers
#predictions = model.predict(X_train)
#for i in range(len(predictions)):
#	cv2.imwrite('./treatments/imgCenterCropped'+str(i)+'.jpg', predictions[i]*255)