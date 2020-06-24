import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np

print("Importing Data from HDF5...")
f = h5py.File('Images_Labels_color_keras.hdf5', 'r')
images = f['images']
labels = f['labels']
labels = [n.decode('utf8') for n in labels]

#normaling images in case we decide to use colored images
images = [n for n in images]
data = np.array(images, dtype='float') / 255.0


print("One Hot Encoding the labels...")
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 36)


print("Test/Train Spliting...")
(train_X, test_X, train_Y, test_Y) = train_test_split(data, 
						      labels, 
						      test_size=0.25, 
						      random_state=42, 
						      shuffle=True)



#can use either grid search cv from scikit-learn and wrap it around tf, or use various for loops to optimize following hyperparamters
print("Compiling Model...")
start_time = int(time.time())
dense_layers = 1
layer_sizes = 32
conv_layers = 4
EPOCHS = 50
KERNEL_SIZE = (5, 5)

NAME = "{}-conv-{}-nodes-{}-dense-{}-epochs-{}-kernel-{}".format(
    	conv_layers, 
    	layer_sizes, 
    	dense_layers, 
    	EPOCHS, 
    	KERNEL_SIZE, 
    	int(time.time()))
    tboard_dir = os.path.join("logs", NAME)
    tensorboard = TensorBoard(log_dir=tboard_dir)
    earlystop_callback = EarlyStopping(
          monitor='val_accuracy', min_delta=0.01,
          patience=5)

model = Sequential()

model.add(Convolution2D(layer_sizes, KERNEL_SIZE, input_shape=train_X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(layer_sizes, KERNEL_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(layer_sizes, KERNEL_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(layer_sizes, KERNEL_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(36, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
model.compile(optimizer=opt, 
			  loss='categorical_crossentropy',	
			  metrics=['accuracy'])

print("Augmenting Data...")
train_aug = ImageDataGenerator(
		   rotation_range=20,
		   zoom_range=0.15,
		   width_shift_range=0.2,
		   height_shift_range=0.2,
		   shear_range=0.15,
		   horizontal_flip=True, 
		   )
test_aug = ImageDataGenerator()

train_set = train_aug.flow(x=train_X, y=train_Y, batch_size=layer_sizes)
test_set = test_aug.flow(x=test_X, y=test_Y, batch_size=layer_sizes)




print("Fitting Model to Data...")
history = model.fit(train_set,
				  steps_per_epoch=len(train_X) // layer_sizes,
				  epochs=EPOCHS, 
				  validation_data=test_set,
				  validation_steps=len(test_X) // layer_sizes, 
				  callbacks=[tensorboard])



model.save("final_model_color.h5")
model.save_weights('final_model_color_weights.h5')


#plotting utility
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plots/{}.png".format(NAME))
