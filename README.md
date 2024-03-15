# Autonomous Driving w/ Deep Learning
This project uses behavioral cloning to train a car to drive autonomously in a simulator. The simulator provides images from three cameras mounted on the car, as well as the steering angle, throttle, brake, and speed of the car. The goal is to train a neural network to predict the steering angle based on the images from the three cameras. The neural network is a Convolutional Neural Network trained using Keras and TensorFlow. I would like to thank the TensorFlow Research Cloud for providing the TPU v4-8 used during training.

The simulator can be downloaded from: https://github.com/udacity/self-driving-car-sim

## Data Collection
I used this dataset(all 3 subsets): https://www.kaggle.com/datasets/zaynena/selfdriving-car-simulator

## Model Architecture
The model architecture is based on the NVIDIA model: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

## Checkpoints
A model checkpoint can be downloaded from: https://huggingface.co/sr5434/self-driving-car

Wandb logs: https://wandb.ai/samirrangwalla1/self-driving/runs/nsj7wwer

## Scripts
 - ```main.py``` - The main script to train the model
 - ```drive.py``` - The script to drive the car in the simulator
 - ```dummy_requestor.py``` - A script to test the server(I was unable to get the simulator working on my Apple Silicon Mac, so I used this to test)
