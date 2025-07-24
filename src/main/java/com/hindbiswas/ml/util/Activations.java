package com.hindbiswas.ml.util;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.models.Activation;
import com.hindbiswas.ml.models.BatchActivation;

/**
 * Activations
 */
public class Activations {

    public static Activation sigmoid() {
        return x -> 1.0 / (1.0 + Math.exp(-x));
    }

    public static Activation tanh() {
        return x -> Math.tanh(x);
    }

    public static Activation relu() {
        return x -> Math.max(0, x);
    }

    public static Activation identity() {
        return x -> x;
    }

    public static Activation leakyRelu(double alpha) {
        return x -> Math.max(alpha * x, x);
    }

    public static Activation parametricRelu(double alpha) {
        return x -> Math.max(0, x) + alpha * Math.min(0, x);
    }

    public static Activation elu(double alpha) {
        return x -> x >= 0 ? x : alpha * (Math.exp(x) - 1);
    }

    public static BatchActivation softmax() {
        return (SimpleMatrix z) -> {
            // Subtract max for numerical stability
            SimpleMatrix max = new SimpleMatrix(1, z.getNumCols());
            for (int i = 0; i < z.getNumCols(); i++) {
                max.set(0, i, z.extractVector(false, i).elementMaxAbs());
            }

            SimpleMatrix stabilized = z.copy();
            for (int col = 0; col < z.getNumCols(); col++) {
                for (int row = 0; row < z.getNumRows(); row++) {
                    stabilized.set(row, col, z.get(row, col) - max.get(0, col));
                }
            }

            // Exponentiate
            SimpleMatrix exp = new SimpleMatrix(z.getNumRows(), z.getNumCols());
            for (int i = 0; i < z.getNumRows(); i++) {
                for (int j = 0; j < z.getNumCols(); j++) {
                    exp.set(i, j, Math.exp(stabilized.get(i, j)));
                }
            }

            // Sum over rows (classes)
            SimpleMatrix colSums = new SimpleMatrix(1, z.getNumCols());
            for (int col = 0; col < z.getNumCols(); col++) {
                double sum = 0.0;
                for (int row = 0; row < z.getNumRows(); row++) {
                    sum += exp.get(row, col);
                }
                colSums.set(0, col, sum);
            }

            // Divide by column-wise sums to normalize
            SimpleMatrix softmax = new SimpleMatrix(z.getNumRows(), z.getNumCols());
            for (int i = 0; i < z.getNumRows(); i++) {
                for (int j = 0; j < z.getNumCols(); j++) {
                    softmax.set(i, j, exp.get(i, j) / colSums.get(0, j));
                }
            }

            return softmax;
        };
    }
}
