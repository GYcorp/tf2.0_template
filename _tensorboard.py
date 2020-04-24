
# tf log
generator_loss_metrics = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_loss_metrics = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/train_loss/')

with summary_writer.as_default():
    tf.summary.scalar('disc_loss', disc_loss, step=index)        
    tf.summary.image("test_image", tf.cast(predictions * 127.5 + 127.5, dtype=tf.uint8), step=epoch)
    tf.summary.histogram()
    tf.summary.audio()