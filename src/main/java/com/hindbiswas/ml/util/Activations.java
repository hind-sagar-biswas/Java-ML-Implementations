package com.hindbiswas.ml.util;

import com.hindbiswas.ml.models.Activation;

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
}
