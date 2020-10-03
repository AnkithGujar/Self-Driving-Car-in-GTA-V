# Self Driving Car in the GTA V #

## Description ##

The main focus of our project was to achieve two aspects of self-driving cars, Steering control and Obstacle avoidance. We trained our Convolutional Neural Network (CNN) with images taken from a camera mounted on the vehicle to achieve accurate steering control and we used a pre-trained network on the Common Objects in Context (COCO) dataset for Object Detection. The code in this repository focuses on testing our model in the game GTA V. We picked GTA V for its real-world simulation with traffic, pedestrians, traffic lights, and buildings.

## How to use ##

### Train your own model ###

If you want to train your own model, there are a couple of things you need to do. First run the [collectdata.py](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/blob/master/collectdata.py) script and drive around in the game. The script basically collects the images and controller inputs you give and stores it in a .npy file. 

Once the data is collected, you will need to balance it. 

To train the model on the collected data, run the [train.py](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/blob/master/train.py) script. This script trains and saves the best model in the [models folder](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/tree/master/models). 

Now follow the steps mentioned in the next section to test the model.

### Run pre-trained model ###

There is a pre-trained keras model, in the [models folder](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/tree/master/models), called [model_gtav.h5](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/blob/master/models/model_gtav.h5). To use this model you just need to run the [drive.py](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/blob/master/drive.py) script.

You will need to install the [Xbox 360 Controller Emulator](https://www.x360ce.com/) to make the python script send inputs to the game.

## Dependencies ##

All the libraries and their versions required to run the project are listed in the [environment.yml](https://github.com/AnkithGujar/Self-Driving-Car-in-GTA-V/blob/master/environment.yml) file.

## Refrences ##

Major inspiration for this project was [Sentdex/pygta5](https://github.com/Sentdex/pygta5). Check out his [youtube playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a) as well.

The convolutional neural network architecture used in this project was taken from the paper [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).