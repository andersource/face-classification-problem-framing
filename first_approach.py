from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = fetch_olivetti_faces()
X = data['data']
y = to_categorical(data['target'])

test_accuracies = []
for _ in range(100):
	indices_for_train, indices_for_test = train_test_split(np.arange(10), train_size=2)
	i = np.hstack([[10 * i + j for j in indices_for_train] for i in range(40)])
	X_train = X[i]
	y_train = y[i]
	i = np.hstack([[10 * i + j for j in indices_for_test] for i in range(40)])
	X_test = X[i]
	y_test = y[i]

	model = Sequential([
			Dense(128, input_shape=(X_train.shape[1], ), activation='relu'),
			BatchNormalization(),
			Dense(64, activation='relu'),
			BatchNormalization(),
			Dense(32),
			Dense(y_train.shape[1], activation='softmax')
		])

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(X_train, y_train, epochs=1200, verbose=0)

	test_accuracies.append(accuracy_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1)))

print('Mean test accuracy: %.3f' % np.mean(test_accuracies))
print('Test accuracy std: %.3f' % np.std(test_accuracies))
np.save('performance_results/first_approach', np.array(test_accuracies))

