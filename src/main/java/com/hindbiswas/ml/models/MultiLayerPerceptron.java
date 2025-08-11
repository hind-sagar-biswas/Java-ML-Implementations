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
    private final double learningRate;

    private LossFunction lossFunction = null;

    private int epochs = 1;
    private int batchSize = 1;
    private double validationSplit = 0.0;

    private int hiddenLayersAdded = 0;
    private Layer[] layers;

    private boolean fitted = false;

    public MultiLayerPerceptron(int inputSize, int hiddenLayers, int outputSize) {
        this(inputSize, hiddenLayers, outputSize, 0.01);
    }

    public MultiLayerPerceptron(int inputSize, int hiddenLayers, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        this.layers = new Layer[hiddenLayers + 1];

        this.lossFunction = (pred, label) -> {
            if (pred.getNumRows() != label.getNumRows() || pred.getNumCols() != label.getNumCols()) {
                throw new IllegalArgumentException("Input and output matrices must have the same dimensions.");
            }
            return pred.minus(label);
        };
    }

    public MultiLayerPerceptron layer(int perceptrons, LayerActivation activation) throws IllegalStateException {
        if (hiddenLayersAdded >= layers.length) {
            throw new IllegalStateException("Cannot add more layers.");
        }

        if (hiddenLayersAdded == 0) {
            layers[hiddenLayersAdded] = new Layer(this.inputSize, perceptrons, activation);
        } else {
            int prevIdx = hiddenLayersAdded - 1;
            layers[hiddenLayersAdded] = new Layer(layers[prevIdx].perceptrons, perceptrons, activation);
        }

        hiddenLayersAdded++;
        return this;
    }

    public MultiLayerPerceptron configure(int epochs, int batchSize, double validationSplit) {
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.validationSplit = validationSplit;

        return this;
    }

    public MultiLayerPerceptron loss(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
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

        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < dataX.size(); j++) {
                SimpleMatrix x = Matrix.columnWithoutBias(dataX.get(j));
                SimpleMatrix y = new SimpleMatrix(outputSize, 1);

                double label = dataY.get(j);
                y.set((int) label, 0, 1.0);

                for (Layer layer : layers)
                    x = layer.feedForward(x);

                SimpleMatrix delta = lossFunction.apply(x, y);
                for (int layerIdx = layers.length - 1; layerIdx >= 0; layerIdx--) {
                    SimpleMatrix deltaPrev = layers[layerIdx].backpropagate(delta, learningRate);
                    if (layerIdx > 0) {
                        SimpleMatrix prevDeriv = layers[layerIdx - 1].getActivationDerivativeOfPreActivation();
                        delta = deltaPrev.elementMult(prevDeriv);
                    } else {
                        delta = deltaPrev;
                    }
                }
            }
        }

        this.fitted = true;
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

        SimpleMatrix xMatrix = Matrix.columnWithoutBias(x);
        for (Layer layer : layers) {
            xMatrix = layer.feedForward(xMatrix);
        }

        ArrayList<Double> output = new ArrayList<>();
        for (int i = 0; i < outputSize; i++) {
            output.add(xMatrix.get(i, 0));
        }

        return output;
    }

    public SimpleMatrix predict(SimpleMatrix xMatrix) throws IllegalArgumentException, IllegalStateException {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        if (xMatrix.getNumRows() != inputSize || xMatrix.getNumCols() != 1) {
            throw new IllegalArgumentException(
                    String.format("Expected input size %d, but got %d.", inputSize, xMatrix.getNumRows()));
        }

        for (Layer layer : layers) {
            xMatrix = layer.feedForward(xMatrix);
        }

        SimpleMatrix output = new SimpleMatrix(outputSize, 1);
        for (int i = 0; i < outputSize; i++) {
            output.set(i, 0, xMatrix.get(i, 0));
        }

        return output;
    }
}
