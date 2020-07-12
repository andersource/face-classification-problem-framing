from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input, concatenate
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


def weighted_categorical_crossentropy(weights):
	weights = K.variable(weights)

	def loss(y_true, y_pred):
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		loss = y_true * K.log(y_pred) * weights
		loss = -K.sum(loss, -1)
		return loss

	return loss


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

	X1_train = []  # First face
	X2_train = []  # Second face
	y1_train = []  # First face output
	y2_train = []  # Second face output
	y3_train = []  # Comparison output

	for i in range(pre_X_train.shape[0]):
		for j in range(pre_X_train.shape[0]):
			if i == j:
				continue

			similarity = 0
			if pre_y_train[i].argmax() == pre_y_train[j].argmax():
				similarity = 1

			X1_train.append(pre_X_train[i])
			X2_train.append(pre_X_train[j])
			y1_train.append(pre_y_train[i].argmax())
			y2_train.append(pre_y_train[j].argmax())
			y3_train.append(similarity)

	X1_train = np.array(X1_train)
	X2_train = np.array(X2_train)
	y1_train = to_categorical(y1_train)
	y2_train = to_categorical(y2_train)
	y3_train = to_categorical(y3_train)

	x1 = Input(shape=(pre_X_train.shape[1],), name='face1')
	x2 = Input(shape=(pre_X_train.shape[1],), name='face2')
	L1 = Dense(128, activation='relu', input_shape=(x1.shape[1],), name='face_rep1')
	BN1 = BatchNormalization(name='batch_norm1')
	L2 = Dense(64, activation='relu', input_shape=(128,), name='face_rep2')
	BN2 = BatchNormalization(name='batch_norm2')
	L3 = Dense(32, activation='relu', input_shape=(64,), name='face_rep3')
	O1 = Dense(40, activation='softmax', input_shape=(32,), name='face_class')

	R1 = BN2(L2(BN1(L1(x1))))
	R2 = BN2(L2(BN1(L1(x2))))

	C1 = concatenate([R1, R2], name='face_rep_concat')
	L4 = Dense(64, activation='relu', input_shape=(128,), name='comparison_dense')
	BN3 = BatchNormalization(name='batch_norm3')
	O2 = Dense(2, activation='softmax', input_shape=(64,), name='comparison_res')

	face1_res = O1(L3(R1))
	face2_res = O1(L3(R2))
	comparison_res = O2(BN3(L4(C1)))

	model = Model(inputs=[x1, x2], outputs=[face1_res, face2_res, comparison_res])

	tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0005),
				  loss=[
						tf.keras.losses.categorical_crossentropy,
						tf.keras.losses.categorical_crossentropy,
						weighted_categorical_crossentropy([1, 79]),
				  ],
				  loss_weights=[.1, .1, 1.])

	for i in range(130):
		hist = model.fit([X1_train, X2_train], [y1_train, y2_train, y3_train], epochs=10, verbose=0)
		last_loss = hist.history['comparison_res_loss'][-1]
		lr = .0005
		if last_loss <= .5:
			lr = .0001
		if last_loss <= .1:
			lr = .00001

		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
					  loss=[
						  tf.keras.losses.categorical_crossentropy,
						  tf.keras.losses.categorical_crossentropy,
						  weighted_categorical_crossentropy([1, 39]),
					  ],
					  loss_weights=[.05, .05, 1.])


	def predict_single(model, x):
		X1 = []
		X2 = []
		for i in range(pre_X_train.shape[0]):
			X1.append(x)
			X2.append(pre_X_train[i])

		X1 = np.array(X1)
		X2 = np.array(X2)

		y = model.predict([X1, X2])
		p = y[2][:, 1]
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
np.save('performance_results/combined_approach', np.array(test_accuracies))

