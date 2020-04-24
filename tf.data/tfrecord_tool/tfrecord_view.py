import tensorflow as tf
import numpy as np
import cv2

raw_dataset = tf.data.TFRecordDataset("./name_here.tfrecord")


def _parse_image_function(example_proto):
    parsed_data = tf.io.parse_single_example(example_proto, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'coord_x': tf.io.FixedLenFeature([], tf.float32),
        'coord_y': tf.io.FixedLenFeature([], tf.float32),
        'size': tf.io.FixedLenFeature([], tf.float32),
        'angle': tf.io.FixedLenFeature([], float),
    })
    image = tf.image.decode_image(parsed_data['image'])
    coord_x = parsed_data['coord_x']
    coord_y = parsed_data['coord_y']
    size = parsed_data['size']
    angle = parsed_data['angle']
    return image, coord_x, coord_y, size, angle

parsed_dataset = raw_dataset.map(_parse_image_function)
for image, coord_x, coord_y, size, angle in parsed_dataset.take(1):
    print(image, coord_x, coord_y, size, angle)

# for raw_record in raw_dataset.take(100):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())

#     image_bytes = example.features.feature['image'].bytes_list.value
#     coord_x = example.features.feature['coord_x'].float_list.value
#     coord_y = example.features.feature['coord_y'].float_list.value
#     size = example.features.feature['size'].float_list.value
#     angle = example.features.feature['angle'].float_list.value

#     image = tf.image.decode_image(image_bytes[0])
#     cv2.imshow("image", image.numpy())
#     cv2.waitKey(1)
#     print("\n")
#     for coord_x, coord_y, size, angle in zip(coord_x, coord_y, size, angle):
#         print(coord_x, coord_y, size, angle)