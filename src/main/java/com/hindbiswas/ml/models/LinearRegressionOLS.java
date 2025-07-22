package com.hindbiswas.ml.models;

import java.util.ArrayList;

/**
 * Ordinary Least Squares Linear Regression implementation.
 */
public class LinearRegressionOLS {
    private Double slope;
    private Double intercept;

    private Double meanX;
    private Double meanY;

    // Store internally as ArrayList for efficient random access
    private ArrayList<Double> dataX;
    private ArrayList<Double> dataY;

    /**
     * Fits the model to the provided data. Accepts any List<Double> implementation.
     * 
     * @param dataX List of independent variable values
     * @param dataY List of dependent variable values
     * @return This model instance, for chaining.
     * @throws IllegalArgumentException if sizes differ or lists are empty.
     */
    public LinearRegressionOLS fit(ArrayList<Double> dataX, ArrayList<Double> dataY) throws IllegalArgumentException {
        if (dataX == null || dataY == null || dataX.size() != dataY.size() || dataX.isEmpty()) {
            throw new IllegalArgumentException("Data lists must be non-null, of the same non-zero length.");
        }

        // Copy inputs to internal storage to prevent external mutation
        this.dataX = new ArrayList<>(dataX);
        this.dataY = new ArrayList<>(dataY);

        calculateMean();
        calculateCoefficients();

        return this;
    }

    /**
     * Predicts the target value for a given x.
     * 
     * @param x input value
     * @return predicted y
     * @throws IllegalStateException if fit() has not been called.
     */
    public Double predict(Double x) throws IllegalStateException {
        if (slope == null || intercept == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        return slope * x + intercept;
    }

    public Double getSlope() {
        return slope;
    }

    public Double getIntercept() {
        return intercept;
    }

    @Override
    public String toString() {
        if (slope == null || intercept == null) {
            return "LinearRegressionOLS (unfitted model)";
        }
        return String.format(
                "LinearRegressionOLS [y = %.4fx + %.4f]",
                slope, intercept);
    }

    private void calculateMean() {
        meanX = 0.0;
        meanY = 0.0;
        for (int i = 0; i < dataX.size(); i++) {
            meanX += dataX.get(i);
            meanY += dataY.get(i);
        }
        meanX /= dataX.size();
        meanY /= dataY.size();
    }

    private void calculateCoefficients() {
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < dataX.size(); i++) {
            double dx = dataX.get(i) - meanX;
            numerator += dx * (dataY.get(i) - meanY);
            denominator += dx * dx;
        }
        slope = numerator / denominator;
        intercept = meanY - slope * meanX;
    }
}
