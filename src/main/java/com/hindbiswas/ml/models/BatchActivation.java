package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

/**
 * Functional interface for activation functions.
 */
@FunctionalInterface
public interface BatchActivation {
    /**
     * Applies the activation function to a raw value.
     * 
     * @param x raw input
     * @return predicted label (+1 or -1)
     */
    SimpleMatrix apply(SimpleMatrix x);
}
