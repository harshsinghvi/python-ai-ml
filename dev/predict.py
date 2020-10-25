import tensorflow as tf


model = tf.keras.models.load_model("linear.model")

prediction = model.predict([0,20])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

print(prediction)
# print(type(prediction))

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
