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

        this.weights = Matrix.random(this.perceptrons, inputs + 1, 0);
    }

    public SimpleMatrix feedForward(SimpleMatrix input) {
        this.input = new SimpleMatrix(this.inputs + 1, 1);

        for (int i = 0; i < this.inputs; i++) {
            this.input.set(i, 0, input.get(i, 0));
        }
        this.input.set(this.inputs, 0, 1);

        preActivationOutput = weights.mult(this.input);
        activationOutput = activation.apply(preActivationOutput);

        return activationOutput;
    }

    public SimpleMatrix backpropagate(SimpleMatrix delta, double learningRate, boolean derivate) {
        SimpleMatrix gradW = delta.mult(input.transpose());
        weights = weights.minus(gradW.scale(learningRate));

        SimpleMatrix back = weights.transpose().mult(delta);

        return derivate
                ? back.elementMult(activation.derivative(preActivationOutput))
                : back;
    }
}
