
# useful modules
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint



# read csv file + save to variable: lines;
lines = []
file_dic = "./data/"
with open(file_dic + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# only used for debug.
#samples = []
#for sample in lines:
#        samples.append(sample)




# remove redundancy data, ensures recovery data
angles = []
samples = []
center_counter = 0
for sample in lines:
    center_value = float(sample[3])
#    biased tuning parameters to filter steering data
    if center_value > 0.2 and center_value < -0:
        angles.append(center_value)
        samples.append(sample)
    if center_value <= 0.2 or center_value >= 0:
#        quadruple 0 will leads to removal of angles.
        if center_counter >= 4:
            angles.append(center_value)
            samples.append(sample)
            center_counter = 0
        center_counter += 1

## Plot distribution of samples steering initial angles after prefiltering central angle samples
#plt.hist(samples)
#plt.title("Steering angles Histogram after filtering")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.gcf()
#plt.show()

# split the train set and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# print("Shape of read non-redundant images dataset", np.shape(samples))

# create adjusted steering measurements for the side camera images
# correction radio for the steering angle

def generator(samples, batch_size=32):
    correction = 0.25
    sample_count = len(samples)
    while True:
        shuffle(samples)
#        capsule the data package
        for offset in range(0, sample_count, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images  = []
            angles = []
            X_train = []
#            handling augmented images and angles
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                flipped_center_angle = -1.0 * (center_angle)
#                introduce image from 3 cameras
                for i in range(3):
                    image_addr = batch_sample[i]
                    image_path = image_addr.split('/')[-1]
                    image_path = "./data/IMG/" + image_path
                    image = cv2.imread(image_path)
#                    transforming colorspace
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image = image[70:140,10:310]
                    image = cv2.resize(image,(24,24))
                    images.append(image)
#                    flip the image to augment the samples
                    images.append(cv2.flip(image, 1))
#                    add correction for the left/right camera,
                    if i == 0:
                        angle = center_angle
                        flipped_angle = flipped_center_angle
                    elif i == 1:
                        angle = center_angle + correction
                        flipped_angle = flipped_center_angle -correction
                    else:
                        angle = center_angle - correction
                        flipped_angle = flipped_center_angle + correction
                    
#                   package the output image and angle
                    angles.append(angle)
                    angles.append(flipped_angle)
        
#        shuffle the output

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train,y_train)

model = Sequential()

# input image size trial and error
#row,col,ch = 80,160,3
#row,col,ch = 70,300,3
#row,col,ch = 35,150,3
#row, col, ch = 60,60,3
#row,col,ch = 30,30,3
row,col,ch = 24,24,3


# Create NVIDIA Dave2 covnet with keras
# Normalization
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(row, col, ch)))


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


# implement the training process.validate the dataset loss.
history = model.fit_generator(generator(train_samples, batch_size=32), samples_per_epoch = (len(train_samples)//192)*192*6,
                     nb_epoch=20,validation_data=generator(validation_samples, batch_size=32), nb_val_samples=len(validation_samples)*6)

## Plot cost history
#plt.plot(fit_history.history['loss'])
#plt.plot(fit_history.history['val_loss'])
#plt.title('model MSE loss')
#plt.ylabel('MSE loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()


# save the weights
model.save('model.h5')
print('Trained h5 model saved.')






