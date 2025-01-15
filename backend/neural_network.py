from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)

# Incarcare set de date Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_data = pd.read_csv(url)

iris_data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]

x = iris_data.drop("variety", axis=1).values
y = iris_data["variety"].values

onehot = OneHotEncoder(sparse_output=False)
y = onehot.fit_transform(y.reshape(-1, 1))

#print(x)
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 30 la suta vor fi date de testare

#print(x_train)
#print(y_train)

print(x_test)
print(y_test)

# Straturile retelei neuronale
input_size = 4
hidden_layer = 6
output_size = 3

# Flag pentru a sti daca modelul este antrenat
model_trained = False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):   # transformă un vector de valori intr-un vector de probabilități, unde suma tuturor valorilor este 1
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def calculeaza_mse(y_real, y_pred):
    mse = np.mean(np.sum((y_real - y_pred) ** 2, axis=1))
    return mse


# Initializam ponderile si bias-urile
global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
weights_input_hidden = np.random.uniform(-0.1, 0.1, (input_size, hidden_layer))
weights_hidden_output = np.random.uniform(-0.1, 0.1, (hidden_layer, output_size))

bias_hidden = np.zeros((1, hidden_layer))
bias_output = np.zeros((1, output_size))

# Parametri pentru antrenare
learning_rate = 0.01
epochs = 10000

# Propagarea semnalului inainte
def propagare_inainte(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = softmax(output_layer_input)

    return hidden_layer_output, output_layer_output

# Propagarea erorilor inapoi
def propagare_inapoi(x, y, hidden_layer_output, output_layer_output, weights_hidden_output, learning_rate):
    # 4.1 Se calculează gradienții erorilor pentru neuronii din stratul de ieșire
    output_error = y - output_layer_output 
    output_gradient = output_layer_output * (1 - output_layer_output) * output_error 

    # 4.2 Se actualizează corecțiile ponderilor dintre stratul ascuns și stratul de ieșire
    delta_weights_hidden_output = learning_rate * np.dot(hidden_layer_output.T, output_gradient)
    bias_output_correction = learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

    # 4.3 Se calculează gradienții erorilor pentru neuronii din stratul ascuns
    hidden_error = np.dot(output_gradient, weights_hidden_output.T)  # ∑(δk * wjk)
    hidden_gradient = hidden_layer_output * (1 - hidden_layer_output) * hidden_error  # δj = yj * (1 - yj) * ∑(...)

    # 4.4 Se actualizează corecțiile ponderilor dintre stratul de intrare și stratul ascuns
    delta_weights_input_hidden = learning_rate * np.dot(x.T, hidden_gradient)
    bias_hidden_correction = learning_rate * np.sum(hidden_gradient, axis=0, keepdims=True)

    return delta_weights_input_hidden, bias_hidden_correction, delta_weights_hidden_output, bias_output_correction


def ajustare_ponderi(delta_weights_input_hidden, bias_hidden_correction, delta_weights_hidden_output, bias_output_correction,
                     weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Actualizăm ponderile și bias-urile pentru stratul ascuns și stratul de ieșire
    weights_input_hidden += delta_weights_input_hidden
    bias_hidden += bias_hidden_correction

    weights_hidden_output += delta_weights_hidden_output
    bias_output += bias_output_correction

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def predict(features, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, onehot):
    # Reshape pentru a fi compatibil cu modelul (1 x 4)
    features = np.array(features).reshape(1, -1)

    # Propagare înainte
    hidden_layer_input = np.dot(features, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = softmax(output_layer_input)

    # Indexul clasei prezise
    predicted_index = np.argmax(output_layer_output, axis=1)[0]

    # Eticheta clasei prezise
    predicted_class = onehot.categories_[0][predicted_index]

    return predicted_class


# Bucla de antrenare
losses = []

@app.route('/train', methods=['POST'])
def antreneaza_model():
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, model_trained

    losses = []

    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = propagare_inainte(
            x_train, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
        )

        delta_weights_input_hidden, bias_hidden_correction, delta_weights_hidden_output, bias_output_correction = propagare_inapoi(
            x_train, y_train, hidden_layer_output, output_layer_output, weights_hidden_output, learning_rate
        )

        weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = ajustare_ponderi(
            delta_weights_input_hidden, bias_hidden_correction,
            delta_weights_hidden_output, bias_output_correction,
            weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
        )

        loss = calculeaza_mse(y_train, output_layer_output)
        losses.append(loss)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoca {epoch + 1}/{epochs}, Pierdere: {loss:.4f}")

    # Matricea de ieșire pentru datele de testare
    _, test_output = propagare_inainte(
        x_test, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    )

    # Matricea de ieșire pentru datele de antrenare
    _, train_output = propagare_inainte(
        x_train, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    )

    # Creăm matricea de rezultate pentru setul de testare
    test_results = []
    for i, probabilities in enumerate(test_output):
        predicted_index = np.argmax(probabilities)
        predicted_class = onehot.categories_[0][predicted_index]
        test_results.append({
            "observation": i + 1,
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted_class
        })

    # Creăm matricea de rezultate pentru setul de antrenare
    train_results = []
    for i, probabilities in enumerate(train_output):
        predicted_index = np.argmax(probabilities)
        predicted_class = onehot.categories_[0][predicted_index]
        train_results.append({
            "observation": i + 1,
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted_class
        })

    model_trained = True

    return jsonify({
        "message": "Modelul a fost antrenat cu succes!",
        "train_results": train_results,
        "test_results": test_results
    })

@app.route('/predict', methods=['POST'])
def predictie_iris():
    try:
        global model_trained

        if not model_trained:
            return jsonify({"error": "Modelul nu a fost antrenat înca!"}), 400
        
        data = request.json
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]

        # Realizează predicția
        predicted_class = predict(features, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, onehot)

        # Returnează rezultatul în format JSON
        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/info', methods=['GET'])
def detalii_retea():
    details = {
        "input_size": input_size,
        "hidden_layer1_size": hidden_layer,
        "output_size": output_size
    }
    return jsonify(details)


if __name__ == '__main__':
    app.run(debug=True)

