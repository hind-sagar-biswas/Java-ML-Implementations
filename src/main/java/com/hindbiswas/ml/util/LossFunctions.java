package com.hindbiswas.ml.util;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.models.LossFunction;

/**
 * LossGradients
 */
public class LossFunctions {

    public static String sse() {
        return "sse";
    }

    public static String mse() {
        return "mse";
    }

    public static LossFunction resolve(String name) throws IllegalArgumentException {
        switch (name) {
            case "sse":
                return LossCalculationFunctions.sse();
            case "mse":
                return LossCalculationFunctions.mse();
            default:
                throw new IllegalArgumentException("Unknown loss gradient: " + name);
        }
    }
}

class LossCalculationFunctions {
    public static LossFunction sse() {
        return (pred, label) -> {
            SimpleMatrix diff = pred.minus(label);
            double sampleLoss = 0.0;
            for (int r = 0; r < diff.getNumRows(); r++) {
                double v = diff.get(r, 0);
                sampleLoss += v * v;
            }
            return sampleLoss;
        };
    }

    public static LossFunction mse() {
        return (pred, label) -> {
            SimpleMatrix diff = pred.minus(label);
            double sampleLoss = 0.0;
            for (int r = 0; r < diff.getNumRows(); r++) {
                double v = diff.get(r, 0);
                sampleLoss += v * v;
            }
            return sampleLoss / diff.getNumRows();
        };
    }
}
