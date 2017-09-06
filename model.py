
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
from keras.utils.visualize_util import plot



# read csv file + save to variable: lines;
lines = []
file_dic = "./data/"
with open(file_dic + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# only used for debug.
# samples = []
# for sample in lines:
#        samples.append(sample)




# steering angle hist gram before filtering
angles = []
samples = []
center_counter = 0
for sample in lines:
    center_value = float(sample[3])
    angles.append(center_value)
    samples.append(sample)

plt.hist(angles)
plt.title("Steering angles Histogram before filtering")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.gcf()
plt.show()


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

# steering angle hist gram after filtering
plt.hist(angles)
plt.title("Steering angles Histogram after filtering")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.gcf()
plt.show()

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

def my_resize_function(images):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(images,(24,24))


model = Sequential()

# input image size

row,col,ch = 160,320,3
input_shape = (row,col,ch)


# Create NVIDIA Dave2 covnet with keras
# Cropping
model.add(Cropping2D(cropping=((70,20),(10,10)),input_shape = input_shape))
# Resizing
#model.add(Lambda(lambda x: ktf.image.resize_images(x,(24,24))))
model.add(Lambda(my_resize_function))
# Normalization
model.add(Lambda(lambda x: x/255 - 0.5))


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

# Plot the covnet
plot(model, to_file='model.png')

# use checkpoint to save the weights
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)

# implement the training process.validate the dataset loss.
history = model.fit_generator(generator(train_samples, batch_size=32), samples_per_epoch = (len(train_samples)//192)*192*6,
                     nb_epoch=10,validation_data=generator(validation_samples, batch_size=32), nb_val_samples=len(validation_samples)*6,verbose=0, callbacks=[checkpointer])

# Plot cost history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


print('Trained h5 model saved.')






