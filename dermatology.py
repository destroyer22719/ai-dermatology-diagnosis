import tensorflow as tf
from keras import layers, models, Input

num_classes = 6

model = models.Sequential([
	Input(shape=(12,)),
	layers.Dense(32, activation='relu'),
	layers.Dropout(0.2),
	layers.Dense(64, activation='relu'),
	layers.Dropout(0.2),
	layers.Dense(32, activation='relu'),
	layers.Dropout(0.2),
	layers.Dense(num_classes, activation='softmax')
])

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)	

model.load_weights('dermatology_model.weights.h5')

model.predict()