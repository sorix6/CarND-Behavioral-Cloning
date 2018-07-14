import os
import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras.backend import tf as ktf

from keras.models import Model
import matplotlib.pyplot as plt

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

samples = []
with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



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
    
	return model

	
def converter(x):

    #x has shape (batch, width, height, channels)
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])

images, measurements = imageLoading()
augmented_images, augmented_measurements = augmentData(images, measurements)
	
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './recordings/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = nvidia()


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
		
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

model.save('model_nvidia_generator.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()