package com.hindbiswas.ml.models;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.hindbiswas.ml.data.MLPModelDTO;
import com.hindbiswas.ml.util.LayerActivations;
import com.hindbiswas.ml.util.LossFunctions;
import com.hindbiswas.ml.util.LossGradients;
import com.hindbiswas.ml.util.Matrix;

/**
 * MultiLayerPerceptron
 */
public class MultiLayerPerceptron {
    private final int inputSize;
    private final int hiddenLayers;
    private final int outputSize;
    private final double learningRate;

    private LossGradient lossGradient = null;
    private LossFunction lossFunction = null;

    private String lossGradientName = null;
    private String lossFunctionName = null;

    private int epochs = 1;
    private int batchSize = 1;
    private double validationSplit = 0.0;

    private int hiddenLayersAdded = 0;
    private Layer[] layers;

    private boolean fitted = false;

    public MultiLayerPerceptron(MLPModelDTO dto) {
        this.inputSize = dto.inputSize;
        this.hiddenLayers = dto.hiddenLayers;
        this.outputSize = dto.outputSize;
        this.learningRate = dto.learningRate;
        this.epochs = dto.epochs;
        this.batchSize = dto.batchSize;
        this.validationSplit = dto.validationSplit;
        this.lossGradient = LossGradients.resolve(dto.lossGradientName);
        this.lossFunction = LossFunctions.resolve(dto.lossFunctionName);
        this.lossGradientName = dto.lossGradientName;
        this.lossFunctionName = dto.lossFunctionName;
        this.layers = new Layer[dto.layers.size()];
        this.fitted = true;
        for (int i = 0; i < dto.layers.size(); i++) {
            this.layers[i] = new Layer(dto.layers.get(i));
            this.hiddenLayersAdded++;
        }
    }

    public MultiLayerPerceptron(int inputSize, int hiddenLayers, int outputSize) {
        this(inputSize, hiddenLayers, outputSize, 0.01);
    }

    public MultiLayerPerceptron(int inputSize, int[] perceptrons, String[] activations) {
        this(inputSize, perceptrons, activations, 0.01);
    }

    public MultiLayerPerceptron(int inputSize, int[] perceptrons, String[] activations, double learningRate) {
        this(inputSize, perceptrons.length, perceptrons[perceptrons.length - 1], learningRate);

        if (perceptrons.length != activations.length) {
            throw new IllegalArgumentException("Perceptrons and activations arrays must have the same length.");
        }

        if (perceptrons.length == 0) {
            throw new IllegalArgumentException("At least one hidden layer must be added.");
        }

        for (int i = 0; i < perceptrons.length; i++) {
            this.layer(perceptrons[i], activations[i]);
        }
    }

    public MultiLayerPerceptron(int inputSize, int hiddenLayers, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        this.layers = new Layer[hiddenLayers + 1];

        this.lossGradientName = LossGradients.softmaxCrossEntropy();
        this.lossFunctionName = LossFunctions.sse();

        this.lossGradient = LossGradients.resolve(this.lossGradientName);
        this.lossFunction = LossFunctions.resolve(this.lossFunctionName);
    }

    public MultiLayerPerceptron layer(int perceptrons, String activation)
            throws IllegalStateException, IllegalArgumentException {
        if (hiddenLayersAdded == layers.length) {
            throw new IllegalStateException("Maximum number of layers reached.");
        }

        LayerActivation activationFunction = LayerActivations.resolve(activation);

        if (hiddenLayersAdded == 0) {
            layers[hiddenLayersAdded] = new Layer(this.inputSize, perceptrons, activationFunction);
        } else {
            int prevIdx = hiddenLayersAdded - 1;
            layers[hiddenLayersAdded] = new Layer(layers[prevIdx].perceptrons, perceptrons, activationFunction);
        }

        hiddenLayersAdded++;
        return this;
    }

    public MultiLayerPerceptron configure(int epochs, int batchSize, double validationSplit)
            throws IllegalStateException {
        if (fitted) {
            throw new IllegalStateException("Model has already been fitted.");
        }
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.validationSplit = validationSplit;

        return this;
    }

    public MultiLayerPerceptron lossGradient(String lossGradient)
            throws IllegalArgumentException, IllegalStateException {
        if (fitted) {
            throw new IllegalStateException("Model has already been fitted.");
        }
        this.lossGradient = LossGradients.resolve(lossGradient);
        return this;
    }

    public MultiLayerPerceptron loss(String lossFunction) throws IllegalArgumentException, IllegalStateException {
        if (fitted) {
            throw new IllegalStateException("Model has already been fitted.");
        }
        this.lossFunction = LossFunctions.resolve(lossFunction);
        return this;
    }

    public MultiLayerPerceptron fit(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException, IllegalStateException {
        if (fitted) {
            throw new IllegalStateException("Model has already been fitted.");
        }

        if (hiddenLayersAdded != layers.length) {
            throw new IllegalStateException(
                    "You must add all layers (hidden + output). Expected " + layers.length + " layers, but added "
                            + hiddenLayersAdded);
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

        final int n = dataX.size();

        // Build and shuffle indices once, then split into validation and training sets
        ArrayList<Integer> allIndices = new ArrayList<>();
        for (int i = 0; i < n; i++)
            allIndices.add(i);
        Collections.shuffle(allIndices, new Random());

        int valSize = (int) (n * validationSplit);
        List<Integer> valIndices = allIndices.subList(0, valSize);
        List<Integer> trainIndices = allIndices.subList(valSize, n);

        double bestValLoss = Double.POSITIVE_INFINITY;
        int patience = 5;
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainIndices, new Random());

            for (int start = 0; start < trainIndices.size(); start += batchSize) {
                int end = Math.min(start + batchSize, trainIndices.size());
                int currentBatchSize = end - start;

                SimpleMatrix[] accumGrads = new SimpleMatrix[layers.length];
                for (int li = 0; li < layers.length; li++) {
                    accumGrads[li] = layers[li].zeroGrad();
                }

                // accumulation
                for (int j = start; j < end; j++) {
                    int idx = trainIndices.get(j);

                    SimpleMatrix x = Matrix.columnWithoutBias(dataX.get(idx));
                    SimpleMatrix y = new SimpleMatrix(outputSize, 1);

                    double label = dataY.get(idx);
                    y.set((int) label, 0, 1.0);

                    for (Layer layer : layers)
                        x = layer.feedForward(x);

                    SimpleMatrix delta = lossGradient.apply(x, y);
                    for (int layerIdx = layers.length - 1; layerIdx >= 0; layerIdx--) {
                        SimpleMatrix gradW = layers[layerIdx].gradient(delta);
                        accumGrads[layerIdx] = accumGrads[layerIdx].plus(gradW);

                        SimpleMatrix deltaPrev = layers[layerIdx].backpropagate(delta);

                        if (layerIdx > 0) {
                            SimpleMatrix prevDeriv = layers[layerIdx - 1].getActivationDerivativeOfPreActivation();
                            delta = deltaPrev.elementMult(prevDeriv);
                        } else {
                            delta = deltaPrev;
                        }
                    }
                }

                for (int li = 0; li < layers.length; li++) {
                    layers[li].applyGradient(accumGrads[li], learningRate, currentBatchSize);
                }
            }

            if (valIndices.size() > 0) {
                double totalValLoss = 0.0;
                int correct = 0;
                for (int idx : valIndices) {
                    SimpleMatrix x = Matrix.columnWithoutBias(dataX.get(idx));
                    for (Layer layer : layers)
                        x = layer.feedForward(x);

                    SimpleMatrix y = new SimpleMatrix(outputSize, 1);
                    y.set(dataY.get(idx).intValue(), 0, 1.0);

                    totalValLoss += this.lossFunction.apply(x, y);

                    // accuracy
                    int predIdx = 0;
                    double best = x.get(0, 0);
                    for (int r = 1; r < x.getNumRows(); r++) {
                        if (x.get(r, 0) > best) {
                            best = x.get(r, 0);
                            predIdx = r;
                        }
                    }
                    if (predIdx == dataY.get(idx).intValue())
                        correct++;
                }

                double avgValLoss = totalValLoss / valIndices.size();
                double valAcc = (double) correct / valIndices.size();

                System.out.printf("Epoch %d/%d — val_loss=%.6f val_acc=%.4f\n", epoch + 1, epochs, avgValLoss, valAcc);

                if (avgValLoss < bestValLoss) {
                    bestValLoss = avgValLoss;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= patience) {
                        System.out.println("Early stopping triggered.");
                        break;
                    }
                }
            } else {
                System.out.printf("Epoch %d/%d — no validation set (validationSplit=%.3f)\n", epoch + 1, epochs,
                        validationSplit);
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

    public double score(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException, IllegalStateException {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
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
        int correct = 0;
        for (int i = 0; i < dataX.size(); i++) {
            ArrayList<Double> x = dataX.get(i);
            int actual = dataY.get(i).intValue();
            ArrayList<Double> predicted = predict(x);
            int predictedIdx = 0;
            double best = predicted.get(0);
            for (int j = 1; j < predicted.size(); j++) {
                if (predicted.get(j) > best) {
                    best = predicted.get(j);
                    predictedIdx = j;
                }
            }
            if (predictedIdx == actual) {
                correct++;
            }
        }
        return (double) correct / dataX.size();
    }

    public boolean export(Path path) {
        System.out.println("Exporting model to " + path);
        try {
            String json = toString();
            Files.write(path, json.getBytes(StandardCharsets.UTF_8));
            System.out.println("Model exported to " + path);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to export model to " + path + ": " + e.getMessage());
            System.err.println("Model: " + toString());
            return false;
        }
    }

    public static MultiLayerPerceptron importModel(Path path) throws Exception {
        System.out.println("Importing model from " + path);
        String json = Files.readString(path, StandardCharsets.UTF_8);
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        MLPModelDTO dto = gson.fromJson(json, MLPModelDTO.class);
        System.out.println("Model imported from " + path);
        return new MultiLayerPerceptron(dto);
    }

    public MLPModelDTO toDTO() {
        MLPModelDTO dto = new MLPModelDTO();
        dto.inputSize = inputSize;
        dto.hiddenLayers = layers.length;
        dto.outputSize = outputSize;
        dto.learningRate = learningRate;
        dto.epochs = epochs;
        dto.batchSize = batchSize;
        dto.validationSplit = validationSplit;
        dto.lossGradientName = lossGradientName;
        dto.lossFunctionName = lossFunctionName;
        for (Layer layer : layers) {
            dto.layers.add(layer.toDTO());
        }
        return dto;
    }

    @Override
    public String toString() {
        MLPModelDTO dto = toDTO();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(dto);
    }
}
