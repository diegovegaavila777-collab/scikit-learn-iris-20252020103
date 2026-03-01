from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Crear modelo
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

print("Precisión:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
