import tensorflow as tf


model = tf.keras.models.load_model("models/3x+3/linear.model")

prediction[0] = model.predict([0,20,12,123,23,3,2.3])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
# prediction[1] = model.predics
print(prediction)


# print(type(prediction))

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
