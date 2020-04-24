def dataset():
    def _map_fn(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

    train_ds = tf.data.Dataset.list_files("C:/Users/PC/Downloads/CelebA/Img/img_align_celeba/*.jpg")
    ds = train_ds.shuffle(buffer_size=4096)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=2)
    return ds