# 🧠 Neural Networks and Uncertainty Estimation

Welcome to the **Neural Networks and Uncertainty Estimation** repository! This project explores the use of **Multi-Layer Perceptrons (MLPs)** for classification tasks and techniques for estimating uncertainty in predictions.

---

## 📂 **Project Overview**

This repository demonstrates how to build and train **MLPs** using **Scikit-learn** and how to estimate uncertainty in predictions using decision functions and predicted probabilities. It includes:

- **Neural Networks**: Building and visualizing MLPs.
- **Activation Functions**: ReLU and Tanh.
- **Uncertainty Estimation**: Using decision functions and predicted probabilities.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **mglearn**
- **NumPy**
- **Matplotlib**

---

## 📊 **Datasets**

The project uses the following datasets:
- **Moons Dataset**: For binary classification with MLPs.
- **Breast Cancer Dataset**: For evaluating MLP performance.
- **Circles Dataset**: For uncertainty estimation.
- **Iris Dataset**: For multiclass uncertainty estimation.

---

## 🧠 **Key Concepts**

### 1. **Multi-Layer Perceptrons (MLPs)**
- A type of feedforward neural network.
- Uses activation functions like ReLU and Tanh.
- Can model complex decision boundaries.

### 2. **Activation Functions**
- **ReLU**: Rectified Linear Unit, defined as `max(0, x)`.
- **Tanh**: Hyperbolic tangent, outputs values between -1 and 1.

### 3. **Uncertainty Estimation**
- **Decision Function**: Outputs confidence scores for each class.
- **Predicted Probabilities**: Outputs probabilities for each class.

---

## 🚀 **Code Highlights**

### Building an MLP
```python
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
```

### Visualizing Decision Boundaries
```python
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend()
plt.show()
```

### Uncertainty Estimation with Decision Function
```python
decision_scores = gbrt.decision_function(X_test)
print("Decision function scores:\n", decision_scores)
```

### Uncertainty Estimation with Predicted Probabilities
```python
probabilities = gbrt.predict_proba(X_test)
print("Predicted probabilities:\n", probabilities)
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/neural-networks-uncertainty.git
   cd neural-networks-uncertainty
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook neural_networks.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
