import warnings
warnings.filterwarnings('ignore')
from keras.layers import Rescaling
import tensorflow as tf
import matplotlib.pyplot as plt

def get_data(path,seed=123,image_size=(512,512),batch_size=8, augmentation = False):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(path,seed=123,image_size=image_size,batch_size=batch_size)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.5)
        ])

    if augmentation:
        dataset = dataset.map(lambda x,y: (data_augmentation(x),y))
    return dataset

def show_sample(dataset,batch_size=8,num_row=1, class_name=None):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(batch_size):
            ax = plt.subplot(num_row, batch_size//num_row, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if class_name == None:
                plt.title(str(labels[i]))
            else:
                plt.title(class_name[labels[i]])
            plt.axis("off")