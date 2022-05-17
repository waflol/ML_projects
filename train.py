import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import Sequential,Model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,SeparableConv2D,Activation,GlobalAveragePooling2D,Add,Rescaling,Resizing
from keras import backend as k
from tensorflow.keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,TensorBoard


def predictor(backbone,num_classes=1):
    x = backbone.output
    x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512,activation='relu')(x)

    if num_classes == 2 or num_classes==1:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    x= Dropout(0.5)(x)
    outputs = Dense(units,activation=activation)(x)
    return Model(backbone.input,outputs)

def train_on_epoch(x,y,model,loss_fn):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    return loss_value,logits,grads

def val_on_epoch(x,y,model,loss_fn):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    # val_logits = tf.argmax(val_logits, axis=1, output_type=tf.int32)
    return val_logits, loss_value

