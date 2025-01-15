async function predictIris() {
    // Colectăm valorile din câmpurile de input
    const sepalLength = parseFloat(document.getElementById('sepal-length').value);
    const sepalWidth = parseFloat(document.getElementById('sepal-width').value);
    const petalLength = parseFloat(document.getElementById('petal-length').value);
    const petalWidth = parseFloat(document.getElementById('petal-width').value);

    try {
        // Trimitere cerere POST la server
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sepal_length: sepalLength,
                sepal_width: sepalWidth,
                petal_length: petalLength,
                petal_width: petalWidth
            })
        });

        // Verificăm dacă răspunsul este valid
        if (!response.ok) {
            // Extragem mesajul de eroare din răspuns
            const errorData = await response.json();
            throw new Error(errorData.error || `Eroare necunoscută: ${response.status}`);
        }

        // Procesăm răspunsul serverului
        const data = await response.json();
        document.getElementById('prediction-result').innerText = `Clasa prezisă: ${data.predicted_class}`;
    } catch (error) {
        console.error('Eroare la realizarea predicției:', error);
        document.getElementById('prediction-result').innerText = `Eroare: ${error.message}`;
    }
}


async function getNetworkDetails() {
    const response = await fetch('http://127.0.0.1:5000/info');
    const data = await response.json();

    document.getElementById('network-details').innerHTML = `
        <p>Dimensiunea intrării: ${data.input_size}</p>
        <p>Dimensiunea primului strat ascuns: ${data.hidden_layer1_size}</p>
        <p>Dimensiunea ieșirii: ${data.output_size}</p>
    `;
}

async function trainNetwork() {
    try {
        const response = await fetch('http://127.0.0.1:5000/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ learning_rate: 0.01, epochs: 10000 })
        });

        if (!response.ok) throw new Error(`Eroare API: ${response.status}`);

        const data = await response.json();
        const train_results = data.train_results;
        const test_results = data.test_results;

        // pentru datele de antrenament
        let train_resultsHTML = `
            <h3>Rezultate date antrenament:</h3>
            <table>
                <thead>
                    <tr>
                        <th>Observație</th>
                        <th>Probabilitate Iris-setosa</th>
                        <th>Probabilitate Iris-versicolor</th>
                        <th>Probabilitate Iris-virginica</th>
                        <th>Predicție</th>
                    </tr>
                </thead>
                <tbody>
        `;

        train_results.forEach(train_result => {
            train_resultsHTML += `
                <tr>
                    <td>${train_result.observation}</td>
                    <td>${train_result.probabilities[0].toFixed(4)}</td>
                    <td>${train_result.probabilities[1].toFixed(4)}</td>
                    <td>${train_result.probabilities[2].toFixed(4)}</td>
                    <td>${train_result.predicted_class}</td>
                </tr>
            `;
        });

        train_resultsHTML += '</tbody></table>';
        document.getElementById('train-result').innerHTML = train_resultsHTML;

        // pentru datele de test
        let test_resultsHTML = `
            <h3>Rezultate date test:</h3>
            <table>
                <thead>
                    <tr>
                        <th>Observație</th>
                        <th>Probabilitate Iris-setosa</th>
                        <th>Probabilitate Iris-versicolor</th>
                        <th>Probabilitate Iris-virginica</th>
                        <th>Predicție</th>
                    </tr>
                </thead>
                <tbody>
        `;

        test_results.forEach(test_result => {
            test_resultsHTML += `
                <tr>
                    <td>${test_result.observation}</td>
                    <td>${test_result.probabilities[0].toFixed(4)}</td>
                    <td>${test_result.probabilities[1].toFixed(4)}</td>
                    <td>${test_result.probabilities[2].toFixed(4)}</td>
                    <td>${test_result.predicted_class}</td>
                </tr>
            `;
        });

        test_resultsHTML += '</tbody></table>';
        document.getElementById('test-result').innerHTML = test_resultsHTML;

    } catch (error) {
        console.error('Eroare la antrenarea modelului:', error);
        document.getElementById('train-result').innerHTML = `<p>Eroare: ${error.message}</p>`;
    }
}


