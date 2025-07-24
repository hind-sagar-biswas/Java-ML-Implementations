package com.hindbiswas.ml.models;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.util.Matrix;

/**
 * MultiLayerPerceptron
 */
public class MultiLayerPerceptron {
    private final int inputSize;
    private final int hiddenLayers;
    private final int outputSize;

    private int epochs = 1;
    private int batchSize = 1;
    private double validationSplit = 0.0;

    private int hiddenLayersAdded = 0;
    private Layer[] layers;

    private boolean fitted = false;

    public MultiLayerPerceptron(int inputSize, int hiddenLayers, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers;
        this.outputSize = outputSize;

        this.layers = new Layer[hiddenLayers + 1];
    }

    public MultiLayerPerceptron layer(int perceptrons, Activation activation) throws IllegalStateException {
        if (hiddenLayersAdded == hiddenLayers + 1) {
            throw new IllegalStateException("Cannot add more layers.");
        }

        layers[hiddenLayersAdded++] = new Layer(perceptrons, activation);
        return this;
    }

    public MultiLayerPerceptron layer(int perceptrons, BatchActivation activation) throws IllegalStateException {
        if (hiddenLayersAdded == hiddenLayers + 1) {
            throw new IllegalStateException("Cannot add more layers.");
        }

        layers[hiddenLayersAdded++] = new Layer(perceptrons, activation);
        return this;
    }

    public MultiLayerPerceptron configure(int epochs, int batchSize, double validationSplit) {
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.validationSplit = validationSplit;

        return this;
    }

    public MultiLayerPerceptron fit(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException, IllegalStateException {
        if (hiddenLayersAdded < hiddenLayers) {
            throw new IllegalStateException("Model's layers have not been added yet.");
        }

        if (hiddenLayers == hiddenLayersAdded) {
            throw new IllegalStateException("Model's output layer has not been added yet.");
        }

        if (dataX == null || dataX.get(0).size() != inputSize) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", inputSize,
                            (dataX == null ? 0 : dataX.get(0).size())));
        }

        if (dataY == null || dataY.size() != dataX.size()) {
            throw new IllegalArgumentException(
                    String.format("Expected %d labels, but got %d.", dataX.size(), (dataY == null ? 0 : dataY.size())));
        }

        return this;
    }

    public ArrayList<Double> predict(ArrayList<Double> x) throws IllegalArgumentException, IllegalStateException {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        if (x == null || x.size() != inputSize) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", inputSize, (x == null ? 0 : x.size())));
        }

        SimpleMatrix xMatrix = Matrix.row(x);
        for (Layer layer : layers) {
            xMatrix = layer.feedForward(xMatrix);
        }

        ArrayList<Double> output = new ArrayList<>();
        for (int i = 0; i < outputSize; i++) {
            output.add(xMatrix.get(i, 0));
        }

        return output;
    }
}
