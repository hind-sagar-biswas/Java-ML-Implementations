package com.hindbiswas.ml.dto;

import java.util.ArrayList;

/**
 * DTO for Multi-Layer Perceptron (MLP) model configuration and parameters.
 */
public class MLPModelDTO extends DTO {
    /** Number of input features. */
    public int inputSize;
    /** Number of hidden layers. */
    public int hiddenLayers;
    /** Number of output classes/units. */
    public int outputSize;
    /** Learning rate for training. */
    public double learningRate;

    /** Number of training epochs. */
    public int epochs;
    /** Batch size for training. */
    public int batchSize;
    /** Fraction of data used for validation. */
    public double validationSplit;

    /** Name of the loss gradient function. */
    public String lossGradientName;
    /** Name of the loss function. */
    public String lossFunctionName;

    /** List of layer configurations. */
    public ArrayList<LayerDTO> layers = new ArrayList<>();
}
