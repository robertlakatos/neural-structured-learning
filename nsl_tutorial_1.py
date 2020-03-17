import os
import tensorflow as tf
import neural_structured_learning as nsl

# Prepare data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a base model -- sequential, functional, or subclass.
model = tf.keras.Sequential([
    tf.keras.Input((28, 28), name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
print(model.summary())

# Wrap the model with adversarial regularization.
# https://www.tensorflow.org/neural_structured_learning/api_docs/python/nsl/configs/make_adv_reg_config
adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2,
                                             adv_step_size=0.05)

# https://www.tensorflow.org/neural_structured_learning/api_docs/python/nsl/keras/AdversarialRegularization
adv_model = nsl.keras.AdversarialRegularization(model,
                                                adv_config=adv_config)

# Compile
adv_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# train
adv_model.fit({'feature': x_train, 'label': y_train},
              batch_size=32,
              epochs=5)

# evaluate
adv_model.evaluate({'feature': x_test, 'label': y_test})
