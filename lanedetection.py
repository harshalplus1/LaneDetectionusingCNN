#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2 as cv


def lanemodel():
    model=tf.keras.Sequential([
        #cnn
        tf.keras.layers.BatchNormalization(input_shape=(80,160,3)),
        tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.UpSampling2D((2,2)),
        tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.UpSampling2D((2,2)),
        tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
         tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
         tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.UpSampling2D((2,2)),
        tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu"),
        tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(3,3),padding='valid', strides=(1,1),activation="relu")
    ])
    model.compile(optimizer="Adam",
                loss=tf.keras.losses.mean_squared_error
            )

    return model


def images():
    train_img=pickle.load(open(r"C:\Users\hkshi\Downloads\full_CNN_train.p","rb"))
    labels=pickle.load(open(r"C:\Users\hkshi\Downloads\full_CNN_labels.p","rb"))
    train_img=np.array(train_img)
    labels=np.array(labels)
    labels=labels/255
    return train_img,labels


train_img,labels=images()
X_train, X_test, y_train, y_test=train_test_split(train_img, labels,test_size=0.25)
mdl=lanemodel()
aug=ImageDataGenerator(channel_shift_range=0.25)
aug.fit(X_train)
mdl.fit_generator(aug.flow(X_train,y_train,batch_size=128),steps_per_epoch=len(X_train)/128,epochs=12,verbose=1,validation_data=(X_test,y_test))
mdl.save(r"C:\Users\hkshi\OneDrive\Desktop\LaneDetectionmodel.h5")

# %%
