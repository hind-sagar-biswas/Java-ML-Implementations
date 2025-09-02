package com.hindbiswas.ml.models;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.hindbiswas.ml.data.DataFrame;
import com.hindbiswas.ml.data.DataPoint;
import com.hindbiswas.ml.dto.MLPModelDTO;
import com.hindbiswas.ml.util.LayerActivations;
import com.hindbiswas.ml.util.LossFunctions;
import com.hindbiswas.ml.util.LossGradients;
import com.hindbiswas.ml.util.Matrix;
import com.hindbiswas.ml.util.ModelIO;

/**
 * MultiLayerPerceptron
 *
 * <p>
 * Feed-forward multi-layer perceptron classifier with manual training loop.
 * Supports configuring layers, training via {@link #fit(DataFrame)}, predicting
 * single examples, scoring on a {@link DataFrame}, exporting/importing model
 * configuration, and converting the model to a DTO for serialization.
 * </p>
 *
 * <p>
 * Typical usage:
 * 
 * <pre>
 * MultiLayerPerceptron mlp = new MultiLayerPerceptron(inputSize, hiddenLayers, outputSize, learningRate)
 *         .layer(64, "relu") // add hidden layers and output layer(s)
 *         .layer(32, "relu")
 *         .layer(outputSize, "softmax"); // final layer
 *
 * mlp.configure(epochs, batchSize, validationSplit);
 * mlp.fit(trainingDataFrame);
 * double acc = mlp.score(testDataFrame);
 * </pre>
 * </p>
 */
public class MultiLayerPerceptron implements Model {
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

    /**
     * Construct a model from a DTO (used for import).
     *
     * @param dto DTO containing model architecture and configuration
     */
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

    /**
     * Create a new MLP with default learning rate of 0.01.
     *
     * @param inputSize    number of input features
     * @param hiddenLayers number of hidden layers (does not include output layer)
     * @param outputSize   number of outputs / classes
     */
    public MultiLayerPerceptron(int inputSize, int hiddenLayers, int outputSize) {
        this(inputSize, hiddenLayers, outputSize, 0.01);
    }

    /**
     * Convenience constructor that accepts arrays of perceptron counts and
     * activation names.
     * The last entry of {@code perceptrons} is treated as the final (output) layer
     * size.
     *
     * @param inputSize   number of input features
     * @param perceptrons array of perceptron counts per layer (includes final
     *                    layer)
     * @param activations array of activation names for each layer (same length as
     *                    perceptrons)
     * @throws IllegalArgumentException if arrays lengths mismatch or are empty
     */
    public MultiLayerPerceptron(int inputSize, int[] perceptrons, String[] activations) {
        this(inputSize, perceptrons, activations, 0.01);
    }

    /**
     * Convenience constructor that accepts arrays and a custom learning rate.
     *
     * @param inputSize    number of input features
     * @param perceptrons  array of perceptron counts per layer (includes final
     *                     layer)
     * @param activations  array of activation names for each layer (same length as
     *                     perceptrons)
     * @param learningRate learning rate used during training
     * @throws IllegalArgumentException if arrays lengths mismatch or are empty
     */
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

    /**
     * Core constructor.
     *
     * @param inputSize    number of input features
     * @param hiddenLayers number of hidden layers (not counting output layer)
     * @param outputSize   number of outputs / classes
     * @param learningRate learning rate for gradient updates
     */
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

    /**
     * Add a layer to the model.
     *
     * <p>
     * The first invocation adds a layer that connects from the model
     * {@code inputSize};
     * subsequent invocations connect from the previous layer's size.
     * </p>
     *
     * @param perceptrons number of units in the new layer
     * @param activation  activation function name (resolved via
     *                    {@link LayerActivations})
     * @return this model (for fluent chaining)
     * @throws IllegalStateException    if maximum number of layers has been reached
     * @throws IllegalArgumentException if activation is invalid (resolved by
     *                                  LayerActivations)
     */
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

    /**
     * Configure training hyper-parameters.
     *
     * @param epochs          number of epochs to train
     * @param batchSize       mini-batch size
     * @param validationSplit fraction of data used for validation (0.0 - 1.0)
     * @return this model (for fluent chaining)
     * @throws IllegalStateException if the model has already been fitted
     */
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

    /**
     * Set the loss gradient function by name.
     *
     * @param lossGradient loss gradient name (resolved via {@link LossGradients})
     * @return this model (for fluent chaining)
     * @throws IllegalArgumentException if the name cannot be resolved
     * @throws IllegalStateException    if the model has already been fitted
     */
    public MultiLayerPerceptron lossGradient(String lossGradient)
            throws IllegalArgumentException, IllegalStateException {
        if (fitted) {
            throw new IllegalStateException("Model has already been fitted.");
        }
        this.lossGradient = LossGradients.resolve(lossGradient);
        return this;
    }

    /**
     * Set the loss function by name.
     *
     * @param lossFunction loss function name (resolved via {@link LossFunctions})
     * @return this model (for fluent chaining)
     * @throws IllegalArgumentException if the name cannot be resolved
     * @throws IllegalStateException    if the model has already been fitted
     */
    public MultiLayerPerceptron loss(String lossFunction) throws IllegalArgumentException, IllegalStateException {
        if (fitted) {
            throw new IllegalStateException("Model has already been fitted.");
        }
        this.lossFunction = LossFunctions.resolve(lossFunction);
        return this;
    }

    /**
     * Fit the model to the provided {@link DataFrame}.
     *
     * <p>
     * This method will:
     * </p>
     * <ul>
     * <li>validate model and dataframe compatibility</li>
     * <li>split the dataframe into train/validation using
     * {@link DataFrame#split(int,int,boolean,int)}</li>
     * <li>iterate shuffled mini-batches using
     * {@link DataFrame#iterateBatches(int,int)}</li>
     * <li>perform forward/backpropagation and apply gradients via each
     * {@link Layer}</li>
     * <li>print validation loss/accuracy each epoch and support early stopping</li>
     * </ul>
     *
     * @param df training dataframe (features must match model {@code inputSize})
     * @return this fitted model
     * @throws IllegalArgumentException if the dataframe is invalid or labels out of
     *                                  range
     * @throws IllegalStateException    if required layers have not been added
     * @throws NullPointerException     if {@code df} is null
     */
    @Override
    public MultiLayerPerceptron fit(DataFrame df)
            throws IllegalArgumentException, IllegalStateException, NullPointerException {
        if (hiddenLayersAdded != layers.length) {
            throw new IllegalStateException(
                    "You must add all layers (hidden + output). Expected " + layers.length + " layers, but added "
                            + hiddenLayersAdded);
        }

        if (hiddenLayers == hiddenLayersAdded) {
            throw new IllegalStateException("Model's output layer has not been added yet.");
        }

        df = Objects.requireNonNull(df, "DataFrame is null.");
        if (df.size() == 0) {
            throw new IllegalArgumentException("DataFrame is empty.");
        }
        if (df.featureCount() != inputSize) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but DataFrame has %d.", inputSize, df.featureCount()));
        }

        final int n = df.size();
        final int valSize = (int) (n * validationSplit);
        Random topRng = new Random();
        int baseSeed = topRng.nextInt();

        DataFrame[] parts = df.split(valSize, n - valSize, true, baseSeed);
        DataFrame valDF = parts[0];
        DataFrame trainDF = parts[1];

        double bestValLoss = Double.POSITIVE_INFINITY;
        int patience = 5;
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            int epochSeed = baseSeed + epoch;

            for (DataFrame batchDf : trainDF.iterateBatches(batchSize, epochSeed)) {
                int currentBatchSize = batchDf.size();

                SimpleMatrix[] accumGrads = new SimpleMatrix[layers.length];
                for (int li = 0; li < layers.length; li++) {
                    accumGrads[li] = layers[li].zeroGrad();
                }

                // accumulation
                for (DataPoint dp : batchDf) {
                    SimpleMatrix x = Matrix.columnWithoutBias(dp.features);
                    SimpleMatrix y = new SimpleMatrix(outputSize, 1);

                    int labelInt = (int) dp.label;
                    if (labelInt < 0 || labelInt >= outputSize) {
                        throw new IllegalArgumentException("Label out of [0, outputSize) range: " + labelInt);
                    }
                    y.set(labelInt, 0, 1.0);

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

            if (valDF.size() > 0) {
                double totalValLoss = 0.0;
                int correct = 0;
                for (DataPoint dp : valDF) {
                    SimpleMatrix x = Matrix.columnWithoutBias(dp.features);
                    for (Layer layer : layers)
                        x = layer.feedForward(x);

                    SimpleMatrix y = new SimpleMatrix(outputSize, 1);
                    y.set((int) dp.label, 0, 1.0);

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
                    if (predIdx == (int) dp.label) {
                        correct++;
                    }
                }

                double avgValLoss = totalValLoss / valDF.size();
                double valAcc = (double) correct / valDF.size();

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

    /**
     * Predict the output (raw scores) for a single example represented as an
     * {@link ArrayList}.
     *
     * @param x feature vector as an ArrayList (size must equal {@code inputSize})
     * @return list of length {@code outputSize} with raw output scores
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if the input length is incorrect
     */
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

    /**
     * Predict the output (raw scores) for a single example represented as a
     * primitive array.
     *
     * @param x feature vector (length must equal {@code inputSize})
     * @return list of length {@code outputSize} with raw output scores
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if the input length is incorrect
     */
    @Override
    public ArrayList<Double> predict(double[] x) throws IllegalArgumentException, IllegalStateException {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        if (x == null || x.length != inputSize) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", inputSize, (x == null ? 0 : x.length)));
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

    /**
     * Predict the output (raw scores) for a single example represented as a
     * {@link SimpleMatrix}.
     *
     * @param xMatrix column vector with {@code inputSize} rows and 1 column
     * @return column vector with {@code outputSize} rows and 1 column of raw output
     *         scores
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if the matrix dimensions are incorrect
     */
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

    /**
     * Compute classification accuracy of the model on a given {@link DataFrame}.
     *
     * <p>
     * Accuracy is computed by taking the argmax of the raw outputs for each row and
     * comparing it to the integer label in the dataframe.
     * </p>
     *
     * @param df evaluation dataframe
     * @return accuracy in [0.0, 1.0]
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if the dataframe is null, empty, or has
     *                                  wrong feature count
     */
    @Override
    public double score(DataFrame df)
            throws IllegalArgumentException, IllegalStateException, NullPointerException {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        df = Objects.requireNonNull(df, "DataFrame is null.");
        if (df.size() == 0) {
            throw new IllegalArgumentException("DataFrame is empty.");
        }
        if (df.featureCount() != inputSize) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but DataFrame has %d.", inputSize, df.featureCount()));
        }

        int correct = 0;
        for (DataPoint dp : df) {
            int actual = (int) dp.label;
            ArrayList<Double> predicted = predict(dp.features);

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
        return (double) correct / df.size();
    }

    /**
     * Export the model (DTO JSON) to the given file path.
     *
     * @param path output path
     * @return true on success, false on failure
     */
    public boolean export(Path path) {
        return ModelIO.export(path, this);
    }

    /**
     * Import a model from a JSON file produced by {@link #export(Path)}.
     *
     * @param path path to the JSON file
     * @return new MultiLayerPerceptron constructed from DTO
     * @throws Exception if reading/parsing fails
     */
    public static MultiLayerPerceptron importModel(Path path) throws Exception {
        return ModelIO.importModel(path, MLPModelDTO.class, MultiLayerPerceptron.class);
    }

    /**
     * Convert the model to a serializable DTO.
     *
     * @return MLPModelDTO representing this model
     */
    @Override
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

    /**
     * Return a JSON representation of the model DTO.
     *
     * @return JSON string
     */
    @Override
    public String toString() {
        MLPModelDTO dto = toDTO();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(dto);
    }
}
