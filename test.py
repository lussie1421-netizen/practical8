
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def test_model():

    data = load_iris()

    model = DecisionTreeClassifier()

    model.fit(data.data, data.target)

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert len(prediction) == 1

    print("Testing successful!")
