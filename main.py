from model import CarModel
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import cv2
import keras
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger
import os

# Unset the environment variable
os.environ.pop('MallocStackLogging', None)

#Device check
devices = tf.config.list_physical_devices()
print(devices)

model = CarModel()
dataroot = "./dataset/dataset"
dataroot2 = ".//track1data/track1data"
dataroot3 = "./track2data/track2data"
ckptroot = "./checkpoint"
lr = 1e-4
weight_decay = 1e-5
batch_size = 64

epochs = 8
# Initialize a new W&B run
wandb.init(
    project="self-driving",
    config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "weight_decay": weight_decay
    }
)


def load_data(data_dir):
    """Load training IMG and train validation split"""

    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    return data_df


train = load_data(dataroot)
train2 = load_data(dataroot2)
train3 = load_data(dataroot3)
imgs_train = []  # Images from various camera angles of the car
outputs_train = []  # Steering angles for each image

def augment(dataroot, imgName, angle):
    """Data augmentation."""
    name = dataroot + '/IMG/' + imgName
    current_image = cv2.imread(name)
    if current_image is None:
        print(name)

    current_image = current_image[65:-25, :, :]
    if np.random.rand() < 0.5:
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0

    return current_image, angle


# Generate train split
print("Processing training data...")

#Dataset 1
for i in tqdm(range(len(train.index))):
    img_path = train.iloc[i]['center'].split('\\')[-1]
    img, angle = augment(dataroot, img_path, train.iloc[i]['steering'])
    imgs_train.append(img)
    outputs_train.append(angle)
    #left
    img_path = train.iloc[i]['left'].split('\\')[-1]
    img, angle = augment(dataroot, img_path, train.iloc[i]['steering'] + 0.4)
    imgs_train.append(img)
    outputs_train.append(angle)
    #right
    img_path = train.iloc[i]['right'].split('\\')[-1]
    img, angle = augment(dataroot, img_path, train.iloc[i]['steering']-0.4)
    imgs_train.append(img)
    outputs_train.append(angle)
#Dataset 2
for i in tqdm(range(len(train2.index))):
    img_path = train2.iloc[i]['center'].split('\\')[-1]
    img, angle = augment(dataroot2, img_path, train2.iloc[i]['steering'])
    imgs_train.append(img)
    outputs_train.append(angle)
    #left
    img_path = train2.iloc[i]['left'].split('\\')[-1]
    img, angle = augment(dataroot2, img_path, train2.iloc[i]['steering'] + 0.4)
    imgs_train.append(img)
    outputs_train.append(angle)
    #right
    img_path = train2.iloc[i]['right'].split('\\')[-1]
    img, angle = augment(dataroot2, img_path, train2.iloc[i]['steering']-0.4)
    imgs_train.append(img)
    outputs_train.append(angle)
#Dataset 3
for i in tqdm(range(len(train3.index))):
    img_path = train3.iloc[i]['center'].split('\\')[-1]
    img, angle = augment(dataroot3, img_path, train3.iloc[i]['steering'])
    imgs_train.append(img)
    outputs_train.append(angle)
    #left
    img_path = train3.iloc[i]['left'].split('\\')[-1]
    img, angle = augment(dataroot3, img_path, train3.iloc[i]['steering'] + 0.4)
    imgs_train.append(img)
    outputs_train.append(angle)
    #right
    img_path = train3.iloc[i]['right'].split('\\')[-1]
    img, angle = augment(dataroot3, img_path, train3.iloc[i]['steering']-0.4)
    imgs_train.append(img)
    outputs_train.append(angle)
# Cosine decay LR scheduler
steps_per_epoch = len(imgs_train) // batch_size
boundaries = [steps_per_epoch * 30, steps_per_epoch * 50]
values = [lr, lr * 0.1, lr * 0.1 * 0.1]
scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
# scheduler = keras.optimizers.schedules.CosineDecay(
#    initial_learning_rate=lr,
#    decay_steps=epochs*steps_per_epoch,
#    alpha=0.0
# )
# Use the Adam optimizer and Mean Squared Error loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=scheduler, weight_decay=weight_decay),
    loss=keras.losses.MeanSquaredError()
)
_ = model(tf.random.normal((1, 3, 70, 320)))  # Allow model params to "mold" to the input shape
model.summary()

print("Starting model training")

model.fit(
    np.array(imgs_train),
    np.array(outputs_train),
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[WandbMetricsLogger()]
)
model.save_weights(ckptroot + '/model.weights.h5')