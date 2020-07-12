from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = fetch_olivetti_faces()
X = data['data']
y = data['target']

test_accuracies = []
for _ in range(100):
    indices_for_train, indices_for_test = train_test_split(np.arange(10), train_size=2)
    i = np.hstack([[10 * i + j for j in indices_for_train] for i in range(40)])
    X_train = X[i]
    y_train = y[i]

    i = np.hstack([[10 * i + j for j in indices_for_test] for i in range(40)])
    X_test = X[i]
    y_test = y[i]

    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(X_train, y_train)

    test_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

print('Mean test accuracy: %.3f' % np.mean(test_accuracies))
print('Test accuracy std: %.3f' % np.std(test_accuracies))
np.save('performance_results/baseline', np.array(test_accuracies))
