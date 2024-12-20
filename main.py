import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Загрузка датасета wine
wine = load_wine()
X, y = wine.data, wine.target

# Визуализация первых 16 образцов данных
feature_names = wine.feature_names

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание и обучение нейросети
mlp = MLPClassifier(hidden_layer_sizes=(80,),
                    activation='logistic',
                    alpha=1e-2,
                    solver='adam',
                    tol=1e-3,
                    random_state=0,
                    learning_rate_init=0.001,
                    verbose=True)

mlp.fit(X_train, y_train)

# Построение графика функции потерь
fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("Number of Iterations")
axes.set_ylabel("Loss")
plt.show()

# Предсказания и расчет точности
predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")