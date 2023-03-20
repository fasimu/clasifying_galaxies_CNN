import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
from visualize import visualize_activations

input_data, labels = load_galaxy_data()

# Take a look at the shape of the data.
print(input_data.shape)
print(labels.shape)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.20, random_state=222, stratify=labels, shuffle=True)

# Preprocess the image input
data_generator = ImageDataGenerator(rescale=1./255)
training_set = data_generator.flow(X_train, y_train, batch_size=5)
testing_set = data_generator.flow(X_test, y_test, batch_size=5)

# Building sequantial model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 3)))
# 1st layer
model.add(tf.keras.layers.Conv2D(5, 3, strides=2, padding="valid", activation="relu"))
# 2nd layer
model.add(tf.keras.layers.Conv2D(8, 3, padding="same",activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
# Output layer
model.add(tf.keras.layers.Dense(4, activation="softmax"))
# Compiling model
opt = tf.keras.optimizers.Adam(lr=0.002)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Fitting model
model.fit(training_set, steps_per_epoch=len(training_set)/5, epochs=50, validation_data=testing_set, validation_steps = len(testing_set)/5)

# Results
visualize_activations(model, testing_set)

# Model evaluation
score = model.evaluate_generator(testing_set, 400)
print("Accuracy = ",score[1])