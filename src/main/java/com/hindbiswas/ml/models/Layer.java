package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.dto.LayerDTO;
import com.hindbiswas.ml.util.LayerActivations;
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

    public Layer(LayerDTO dto) {
        this.inputs = dto.inputs;
        this.perceptrons = dto.perceptrons;
        this.activation = LayerActivations.resolve(dto.activationName);

        this.weights = Matrix.fromArray2D(dto.weights);
    }

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

    public SimpleMatrix gradient(SimpleMatrix delta) {
        return delta.mult(input.transpose());
    }

    public SimpleMatrix backpropagate(SimpleMatrix delta, double learningRate) {
        SimpleMatrix gradW = gradient(delta);

        weights = weights.minus(gradW.scale(learningRate));

        SimpleMatrix back = weights.transpose().mult(delta);

        SimpleMatrix backWithoutBias = back.extractMatrix(1, back.getNumRows(), 0, 1);
        return backWithoutBias;
    }

    public SimpleMatrix backpropagate(SimpleMatrix delta) {
        SimpleMatrix back = weights.transpose().mult(delta);
        return back.extractMatrix(1, back.getNumRows(), 0, 1);
    }

    public void applyGradient(SimpleMatrix gradW, double learningRate, int batchSize) {
        if (batchSize <= 0)
            batchSize = 1;
        SimpleMatrix avgUpdate = gradW.scale(learningRate / batchSize);
        weights = weights.minus(avgUpdate);
    }

    public SimpleMatrix zeroGrad() {
        return new SimpleMatrix(weights.getNumRows(), weights.getNumCols());
    }

    public SimpleMatrix getActivationDerivativeOfPreActivation() {
        return activation.derivative(preActivationOutput);
    }

    public LayerDTO toDTO() {
        LayerDTO dto = new LayerDTO();
        dto.activationName = activation.toString();
        dto.weights = Matrix.toArray2D(weights);
        dto.perceptrons = perceptrons;
        dto.inputs = inputs;
        return dto;
    }
}
