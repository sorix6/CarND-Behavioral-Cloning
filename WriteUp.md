**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

My project includes the following files:
* clone.py - contains the script to create and train the model
* plot.py - contains the same script to create and train the model by using a generator. 
The file was also used to print the training and validation loss metrics
* model_nvidia.h5 - contains a trained convolution neural network using the network proposed by the NVIDIA team
* WriteUp.md - summary of the obtained results
* run1.mp4 - video of the simulator driving the car in autonomous mode

### Model Architecture and Training Strategy

[Model architecture]: https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/model.jpg

After setting up and testing both the LeNet and the NVIDIA architecture, I have concluded that the second one is a better choice for this project.
No dropout layers have been added to the model since I did not feel as it was overfitting.
The number of EPOCHs has been set to 5. as seen in the picture below, the validation loss decreases for every epoch.

https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/loss.jpg

https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/graph.jpg

The model uses an Adam optimizer.

The only changes that have been applied to the architecture are the addition of 3 Lambda layers before the first layer:
* model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3))) - image normalization
* model.add(Lambda(converter)) - calling a converter method that converts an image to grayscale
* model.add(Cropping2D(cropping=((70,25), (0,0)))) - image cropping to remove irrelevant top and bottom parts

[Original image]: https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/imgCenter1-0.jpg "Original"
[After normalization]: https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/imgCenterNormalized0.jpg "After normalization"
[After grayscaling]: https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/imgCenterGrayscale0.jpg "After grayscaling"
[After cropping]: https://raw.githubusercontent.com/sorix6/CarND-Behavioral-Cloning/master/img/imgCenterCropped0.jpg "After cropping"


###Training data

The collection of data has been done through manual driving on the first circuit as follows:
* 2 laps of normal, center focused driving
* 1 lap of driving focused on smooth passing of corners
* 1 lap of driving in the opposite direction (center focused)
* 1 lap of focusing on recovering (only recording the return from the side of the road towards the center)


###Results

As seen in the video run1.mp4, the car is able to complete the full circuit without going off the driveable portion of the road.