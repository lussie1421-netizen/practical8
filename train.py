
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

data = load_iris()

X = data.data
y = data.target

model = DecisionTreeClassifier()

model.fit(X, y)

print("Model trained successfully!")
