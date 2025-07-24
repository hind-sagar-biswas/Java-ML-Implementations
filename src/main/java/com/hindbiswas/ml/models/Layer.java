package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.util.Matrix;

/**
 * Layer
 */
class Layer {
    private final Activation activation;
    private final BatchActivation batchActivation;
    private final int perceptrons;
    private final SimpleMatrix weights;

    public Layer(int perceptrons, Activation activation) {
        this.perceptrons = perceptrons;
        this.activation = activation;
        this.batchActivation = null;

        this.weights = Matrix.random(perceptrons, perceptrons + 1, 0);
    }

    public Layer(int perceptrons, BatchActivation batchActivation) {
        this.perceptrons = perceptrons;
        this.activation = null;
        this.batchActivation = batchActivation;

        this.weights = Matrix.random(perceptrons, perceptrons + 1, 0);
    }

    public SimpleMatrix feedForward(SimpleMatrix input) {
        SimpleMatrix raw = input.mult(weights);

        if (activation == null) {
            return batchActivation.apply(raw);
        }

        SimpleMatrix output = new SimpleMatrix(perceptrons, 1);
        for (int i = 0; i < perceptrons; i++) {
            output.set(i, 0, activation.apply(raw.get(i, 0)));
        }

        return output;
    }
}
