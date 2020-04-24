import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib
data_root = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', fname='flower_photos', untar=True)

data_root = pathlib.Path(data_root)
print(data_root)

for item in data_root.iterdir():
    print(item)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

attributions = (data_root/"LICENSE.txt").read_text(encoding="utf8").splitlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# img_path = all_image_paths[0]
# img_raw = tf.io.read_file(img_path)
# print(repr(img_raw)[:100]+'...')
# img_tensor = tf.image.decode_image(img_raw)
# img_final = tf.image.resize(img_tensor, [192, 192])
# img_final = img_final/255.0

# print(img_final.shape)
# print(img_final.dtype)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# print(path_ds)
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label
image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

BATCH_SIZE = 8

# ds = image_label_ds.shuffle(buffer_size=image_count)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
# ds = ds.prefetch(buffer_size=AUTOTUNE)
# print(ds)

ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds
print(ds)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False

def change_range(image, label):
    return 2*image-1, label
keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))

# feature_map_batch = mobile_net(image_batch)
# print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names))
])

# logit_batch = model(image_batch).numpy()

# print("min logit:", logit_batch.min())
# print("max logit:", logit_batch.max())
# print("shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
loss='sparse_categorical_crossentropy',
metrics=["accuracy"])

model.summary()

model.fit(ds, epochs=2, steps_per_epoch=4)