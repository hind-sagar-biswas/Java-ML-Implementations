package com.hindbiswas.ml.models;

/**
 * Functional interface for activation functions.
 */
@FunctionalInterface
interface Activation {
    /**
     * Applies the activation function to a raw value.
     * 
     * @param x raw input
     * @return predicted label (+1 or -1)
     */
    int apply(double x);
}
