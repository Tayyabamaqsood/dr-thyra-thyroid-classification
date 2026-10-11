# Dr. Thyra, Thyroid Classification

Dr. Thyra is a machine learning project that uses a trained Perceptron model to classify clinical features related to thyroid conditions. The application uses Flask to accept input through a web form, preprocess the data with a saved `StandardScaler`, and display the model's predicted class.

## Project Overview

The project demonstrates the integration of a machine learning classification model with a Flask web application. It uses clinical features, including demographic information, thyroid-related history, and laboratory test measurements.

**Important:** This project is for educational and machine learning demonstration purposes. Its predictions are not a medical diagnosis.

## Features

- Trained Perceptron classification model.
- StandardScaler preprocessing.
- Flask-based prediction endpoint.
- Handling of unmeasured laboratory tests using training-set averages.
- Model inspection and test-case script.

## Technology Stack

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- Joblib
- HTML for the web interface

## Project Structure

```text
dr-thyra-thyroid-classification/
├── app(1).py
├── Thyroid.pkl
├── scaler.pkl
├── check_model.py
├── templates/
│   └── index.html
└── README.md
```

This is the expected structure based on the available files. Include `templates/index.html` only if it exists in your project folder.

## Input Features

The model uses 25 features, in the order recorded by the saved scaler:

- Age and sex
- Thyroxine and antithyroid medication history
- Relevant medical history and thyroid-related indicators
- Laboratory measurement flags and available values for TSH, T3, TT4, T4U, and FTI

The exact feature names and ordering are determined by the saved scaler.

## How It Works

1. The Flask application loads `Thyroid.pkl` and `scaler.pkl`.
2. The application obtains the expected feature order from the scaler.
3. The user submits clinical feature values through the web form.
4. If a supported laboratory test was not measured, the application substitutes that feature's training-set average.
5. The input is transformed using the saved scaler.
6. The Perceptron model generates a prediction.
7. The application displays either `Class 0 (Negative)` or `Class 1 (Positive)`.

These class labels are the model's output labels; their clinical meaning should be confirmed against the original dataset and training process.

## Getting Started

### Prerequisites

- Python installed on your system
- pip
- The saved model and scaler files

### Install Dependencies

```bash
pip install flask pandas numpy scikit-learn joblib
```

### Run the Application

Ensure that `app(1).py`, `Thyroid.pkl`, and `scaler.pkl` are in the same directory.

Run:

```bash
python "app(1).py"
```

The application is configured to run on port `5000`.

Open the local address in your browser:

```text
http://127.0.0.1:5000
```

The HTML template must be present in the expected `templates` directory for the home page to render.

## Model Inspection

The project includes `check_model.py`, a utility that inspects the saved model and scaler, checks feature ordering when the scaler contains feature names, and evaluates several example inputs.

Run it with:

```bash
python check_model.py
```

The script's sample predictions are diagnostic examples, not a substitute for formal model evaluation.

## Limitations

- The original training notebook and dataset are not included in the files reviewed.
- Verified accuracy, precision, recall, and F1-score results are not currently documented.
- The meaning of the predicted classes depends on the original dataset's label mapping.
- Predictions depend on appropriate input values and the same preprocessing used during training.
- The Flask application currently enables debug mode and should be configured appropriately before deployment.

## Future Improvements

- Add the original training notebook and dataset documentation, where licensing permits.
- Document the target labels and verified evaluation metrics.
- Add input validation and automated tests.
- Improve error handling and application security.
- Deploy the application in a suitable environment after validation.

## Author

**Tayyaba Maqsood**

Machine Learning | Python | Flask

## Disclaimer

Dr. Thyra is an educational machine learning project, not a clinically validated diagnostic tool. Its predictions must not be used as the sole basis for medical decisions.
