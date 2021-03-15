import os
import tensorflow as tf
import numpy as np
import utils.config as config

def train_valid_generator(
    IMAGE_SIZE = config.IMAGE_SIZE[:-1],
    BATCH_SIZE = config.BATCH_SIZE,
    data_dir = config.DATA_DIR,
    data_augmentation = config.AUGMENTATION):
    datagen_kwargs = dict(
        rescale=1./255,
        validation_split=0.20
        )
        
    dataflow_kwargs = dict(
        target_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        interpolation = "bilinear"
        )
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    valid_generator = valid_datagen.flow_from_directory(
        directory=data_dir,
        subset="validation", 
        shuffle=False,
        **dataflow_kwargs
    )

    if data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            **datagen_kwargs
        )
    else:
        train_datagen = valid_datagen


    #Training our data for validation
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        subset="training",
        shuffle=True,
        **dataflow_kwargs
    )
    return  train_generator, valid_generator

def manage_input_data(input_image):

    # Convert the input array into dimension
    images = input_image
    size = config.IMAGE_SIZE[:-1]
    resized_input_image = tf.image.resize(
        images,
        size,
        preserve_aspect_ratio=False
        )

    # Datapreparation image
    final_img = np.expand_dims(resized_input_image, axis=0)
    return final_img      

