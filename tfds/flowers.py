import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 224

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="tf_flowers", with_info=True, split=list(splits), as_supervised=True)

get_label_name = metadata.features['label'].int2str
NUM_CLASSES = metadata.features['label'].num_classes

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example).batch(64)
validation = raw_validation.map(format_example).batch(64)
test = raw_test.map(format_example).batch(64)

# plt.figure(figsize=(12,12)) 

# for batch in train.take(1):
#     print(batch)
#     for i in range(9):
#         image, label = batch[0][i], batch[1][i]
#         plt.subplot(3, 3, i+1)
#         plt.imshow(image.numpy())
#         plt.title(get_label_name(label.numpy()))
#         plt.grid(False)    

import cv2
for i, (images, labels) in enumerate(train):
    for image, label in zip(images, labels):
        print(get_label_name(label))
        cv2.imshow('frame', tf.cast((image*255), tf.uint8).numpy())
        if cv2.waitKey(0) == ord('q'): break
    if cv2.waitKey(0) == ord('q'): break