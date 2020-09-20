# Self Driving Car in the GTA V

## Dependencies:

All the software libraries required to run the project is listed in the YML file 'environment.yml'.

## Description:

The main focus of our project was to achieve two major aspects of self-driving cars, Steering control and Obstacle avoidance. We trained our Convolutional Neural Network (CNN) with images taken from a camera mounted on the vehicle to achieve accurate steering control and we used a pre-trained network on the Common Objects in Context (COCO) dataset for Object Detection. The code in this repository focuses on testing our model in the game GTA V. We picked GTA V for its real-world simulation with traffic, pedestrians, traffic lights, and buildings.

## Collecting Data:

To collect the data and create the dataset, a human must drive the vehicle and real-time data in the form of images and their corresponding steering angle controlled by the driver must be collected.

collect_data.py captures an image of the game window and stores the joystick control value which is controlled by the human at that instance. Our data is ultimately a CSV file which maps the image name to its corresponding desired steering angle and throttle value of the vehicle at a particular instance of time.

collect.py relies on 3 main python files: grabscreen.py,getjoy.py, and getkeys.py. The functionalities of each of these files are listed below.
* grabscreen.py - Code to capture images of a specified portion of the screen which is to be used as input to the neural network.
* getjoy.py - Code to collect joystick controls.
* getkeys.py - Code to collect keyboard inputs for manual overides.

Since GTA V accepts only joystick controls, vjoy.py was used to simulate joystick controls from the keyboard keys. The x360ce software is required to be installed for the vjoy.py code to run.

## Balancing Data:

The initial dataset collected was highly biased to driving straight. The data had to be balanced to get an equal number of data points for each class (left, straight, and right). balance.py takes care of this task.

## Network:

For steering control, we trained a Convolutional Neural Network (CNN) which was built on the network used in [this paper](https://arxiv.org/abs/1604.07316).

Object detection was implemented by a pre-trained Yolo V2 network.