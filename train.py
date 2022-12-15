import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False

def unetDeep():
    input_shape=(256,256,3)
    n_channels = input_shape[-1]
    inputs = tf.keras.layers.Input(input_shape)

    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = tf.keras.layers.Dropout(0.2)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = tf.keras.layers.Dropout(0.2)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = tf.keras.layers.Dropout(0.2)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
   
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.2)(conv4)
    
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5')(conv5)
    drop5 = tf.keras.layers.Dropout(0.2)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    drop6 = tf.keras.layers.Dropout(0.2)(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop6))
    merge7 = tf.keras.layers.concatenate([drop3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    drop7 = tf.keras.layers.Dropout(0.2)(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop7))
    merge8 = tf.keras.layers.concatenate([drop2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    drop8 = tf.keras.layers.Dropout(0.2)(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop8))
    merge9 = tf.keras.layers.concatenate([drop1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    conv9 = tf.keras.layers.Conv2D(n_channels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
 
    model = tf.keras.Model(inputs, conv9)

    return model

def myloss(true, prediction):
    
    L1loss = tf.abs(true-prediction)
    L2loss = tf.square(true-prediction)
    return L1loss+100*L2loss
    
if __name__ == "__main__" :

    TRAIN_PATH = './Train'

    x_train = []
    y_train = []

    file_list = os.listdir(TRAIN_PATH)
    file_list.sort()
    for filename in file_list:
        filepath = TRAIN_PATH +  '/' + filename
        print(filename)
        image = imread(filepath)
        image.tolist()
        image = image / 255.
        x_train.append(image)
    
    train_datagen = ImageDataGenerator(
  
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True)
    
    x_train = np.array(x_train)

    x_train = x_train.reshape(-1,256,256,3)

    model = unetDeep()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4), loss=myloss)

    
    checkpoint_path = "model/data-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)


    model.fit_generator(train_datagen.flow(x_train, x_train, batch_size=16), epochs=100,callbacks=[cp_callback])
    