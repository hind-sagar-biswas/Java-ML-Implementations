package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.util.Matrix;

/**
 * Layer
 */
class Layer {
    private final LayerActivation activation;
    private final int inputs;
    public final int perceptrons;

    private SimpleMatrix preActivationOutput;
    private SimpleMatrix activationOutput;
    private SimpleMatrix weights;
    private SimpleMatrix input;

    public Layer(int inputs, int perceptrons, LayerActivation activation) {
        this.inputs = inputs;
        this.perceptrons = perceptrons;
        this.activation = activation;

        this.weights = Matrix.xavier(this.perceptrons, inputs + 1);
    }

    public SimpleMatrix feedForward(SimpleMatrix input) {
        this.input = new SimpleMatrix(this.inputs + 1, 1);
        this.input.set(0, 0, 1);
        for (int i = 0; i < this.inputs; i++) {
            this.input.set(i + 1, 0, input.get(i, 0));
        }

        preActivationOutput = weights.mult(this.input);
        activationOutput = activation.apply(preActivationOutput);

        return activationOutput;
    }

    public SimpleMatrix backpropagate(SimpleMatrix delta, double learningRate) {
        // grad for this layer (perceptrons x (inputs+1))
        SimpleMatrix gradW = delta.mult(input.transpose());
        // update weights including bias column (column 0)
        weights = weights.minus(gradW.scale(learningRate));

        // compute W^T * delta -> size (inputs+1) x 1
        SimpleMatrix back = weights.transpose().mult(delta);

        // drop the bias element (row 0) before sending delta to previous layer
        SimpleMatrix backWithoutBias = back.extractMatrix(1, back.getNumRows(), 0, 1);
        return backWithoutBias; // size inputs x 1
    }

    public SimpleMatrix getActivationDerivativeOfPreActivation() {
        return activation.derivative(preActivationOutput);
    }
}
