
# useful modules
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
# from sklearn.model_selection import train_test_split


# read in the csv file and save the data inside variable lines;
lines = []
file_dic = "./data/"
with open(file_dic + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# remove redundancy data for which steering angle continuously goes with 0 is helpless to train a good model.
angles = []
samples = []
center_counter = 0
for sample in lines:
    center_value = float(sample[3])
    if center_value > 0.01 and center_value < -0.01:
        angles.append(center_value)
        samples.append(sample)
    if center_value <= 0.01 or center_value >= 0.01:
        if center_counter >= 4:
            angles.append(center_value)
            samples.append(sample)
            center_counter = 0
        center_counter += 1

# print("Shape of read non-redundant images dataset", np.shape(samples))

# create adjusted steering measurements for the side camera images
# correction radio for the steering angle

#def generator_cameras(samples, batch_size = 32):


def generator(samples, batch_size=32):
    correction = 0.2
    images = []
    yaws = []
    sample_count = len(samples)
    counter = 0
    while True:
        shuffle(samples)
        for offset in range(0, sample_count, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            input_image  = []
            output_angle = []
            for batch_sample in batch_samples:
                for i in range(3):
                    image_addr = batch_sample[i]
                    image_path = image_addr.split('/')[-1]
                    image_path = "./data/IMG/" + image_path
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image,(160,80))
                    images.append(image)
                    images.append(cv2.flip(image, 1))
        
                yaw = float(batch_sample[3])
                yaws.append(yaw)
                yaws.append(-1.0 * yaw)
                yaws.append(yaw + correction)
                yaws.append(-1.0 * (yaw + correction))
                yaws.append(yaw - correction)
                yaws.append(-1.0 * (yaw - correction))
        
            print("X_train", np.shape(X_train))
            print("y_train", np.shape(y_train))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size= 32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()


row,col,ch = 80,160,3


# Create NVIDIA Dave2 covnet with keras
# Normalization
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((35,10),(5,5))))
# Add 3 * 3x3 convolution layers (output depth 16, 32, and 64), each with ReLU and 2x2 maxpooling layer.
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add a flatten layer
model.add(Flatten())
# 4 Fully connected layers of 400, 100, 20 and 1
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
# Train Covnet
model.compile(loss='mse', optimizer='adam')



model.fit(X_train,y_train,validation_split = 0.2,shuffle = True,nb_epoch = 2)
model.save('model.h5')
print('Trained model saved...')






