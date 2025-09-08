# JavaML Library

JavaML is a lightweight Java library implementing common machine learning models and utilities using core algorithms and the EJML library for matrix operations.

## Models

* **Linear Regression (Ordinary Least Squares)** (`LinearRegressionOLS`)
* **Linear Regression (Batch Gradient Descent)** (`LinearRegressionGD`)
* **Multivariate Linear Regression (OLS)** (`LinearRegressionMultiVar`)
* **Logistic Regression (Gradient Descent)** (`LogisticRegression`)
* **Perceptron** (`Perceptron`)
* **Multi Layer Perceptron** (`MultiLayerPerceptron`)
* **Gaussian Naive Bayes** (`GaussianNB`)
* **Multinomial Naive Bayes** (`MultinomialNB`)
* **Bernoulli Naive Bayes** (`BernoulliNB`)

## Data Structures

* **Data Frame** (`DataFrame`)
* **Data Point** (`DataPoint`)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/hind-sagar-biswas/JavaML
   cd "JavaML"
   ```

2. **Build with Maven**

   ```bash
   mvn clean install
   ```

---

## Example Usage

```java
// Example: Train and evaluate a Perceptron
DataFrame train = ...; // Load or create training data
DataFrame test = ...;  // Load or create test data

Perceptron model = new Perceptron(0.01, 1000, 0.0)
    .shuffle(true)
    .verbose(true)
    .randomizeWeights(42);

model.fit(train);
double acc = model.score(test);
System.out.println("Test accuracy: " + acc);
```

---

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/xyz`)
3. Commit your changes
4. Submit a Pull Request

---

## Author

Hind Biswas  

GitHub: https://github.com/hind-sagar-biswas  

Portfolio: https://hindbiswas.com  

Email: me@hindbiswas.com

---

## License

MIT © Hind Biswas 2025

