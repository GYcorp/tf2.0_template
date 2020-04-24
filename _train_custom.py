# train
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as gradient_tape:
        predictions = model(images, training=True)

        loss = cross_entropy(labels, predictions)

    gradients = gradient_tape.gradient(loss, generator.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss