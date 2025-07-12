package com.hindbiswas.ml.models;

/**
 * LinearRegression
 */
public class LinearRegression {
    public static LinearRegressionGD gradient(Double learningRate, Integer iterations) {
        return new LinearRegressionGD(learningRate, iterations);
    }

    public static LinearRegressionOLS ols() {
        return new LinearRegressionOLS();
    }

    public static LinearRegressionMultiVar multi() {
        return new LinearRegressionMultiVar();
    }
}
