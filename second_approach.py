from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = fetch_olivetti_faces()
X = data['data']
y = to_categorical(data['target'])

test_accuracies = []
for _ in range(100):
	indices_for_train, indices_for_test = train_test_split(np.arange(10), train_size=2)
	i = np.hstack([[10 * i + j for j in indices_for_train] for i in range(40)])
	pre_X_train = X[i]
	pre_y_train = y[i]
	i = np.hstack([[10 * i + j for j in indices_for_test] for i in range(40)])
	pre_X_test = X[i]
	pre_y_test = y[i]

	X_train = []
	y_train = []

	for i in range(pre_X_train.shape[0]):
		for j in range(pre_X_train.shape[0]):
			if i == j:
				continue

			similarity = 0
			if pre_y_train[i].argmax() == pre_y_train[j].argmax():
				similarity = 1

			X_train.append(np.hstack([pre_X_train[i], pre_X_train[j]]))
			y_train.append(similarity)

	X_train = np.array(X_train)
	y_train = np.array(y_train)

	model = Sequential([
			Dense(256, input_shape=(X_train.shape[1], ), activation='relu'),
			BatchNormalization(),
			Dense(128, activation='relu'),
			BatchNormalization(),
			Dense(64),
			BatchNormalization(),
			Dense(2, activation='softmax')
		])

	model.compile(loss='categorical_crossentropy',
				  optimizer=tf.keras.optimizers.Adam(learning_rate=.0001))

	for i in range(45):
		hist = model.fit(X_train, to_categorical(y_train), epochs=10, class_weight={0: 1, 1: 79}, verbose=0)
		last_loss = hist.history['loss'][-1]
		lr = .0001
		if last_loss <= .1:
			lr = .00001
		model.compile(loss='categorical_crossentropy',
					  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))


	def predict_single(model, x):
		X = []
		for i in range(pre_X_train.shape[0]):
			X.append(np.hstack([x, pre_X_train[i]]))

		X = np.array(X)
		p = model.predict(X)[:, 1]
		res = np.zeros(pre_y_train.shape[1])
		for i in range(pre_X_train.shape[0]):
			k = pre_y_train[i].argmax()
			res[k] = max(res[k], p[i])

		return res


	def predict(model, X):
		return np.array([predict_single(model, X[i]) for i in range(X.shape[0])])


	y_train_pred = predict(model, pre_X_train)
	y_test_pred = predict(model, pre_X_test)

	test_accuracies.append(accuracy_score(pre_y_test.argmax(axis=1), y_test_pred.argmax(axis=1)))

print('Mean test accuracy: %.3f' % np.mean(test_accuracies))
print('Test accuracy std: %.3f' % np.std(test_accuracies))
np.save('performance_results/second_approach', np.array(test_accuracies))

