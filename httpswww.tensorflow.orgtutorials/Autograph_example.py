import tensorflow as tf 
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

@tf.function
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

train_dataset = mnist_dataset()

model = tf.keras.Sequential([
    (
        tf.keras.Sequential
        ((
            tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28,28)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu')
        ))
    ),
    (
        tf.keras.Sequential
        ((
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10)
        ))
    )
])

model.build()

# model.summary()

optimizer = tf.keras.optimizers.Adam()

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss

def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if tf.equal(step % 10, 0):
            tf.print('Step', step, ': loss', loss, ': accuracy', compute_accuracy.result())
    return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)
tf.print('Final Step', step, ': loss', loss, ': accuracy', compute_accuracy.result())