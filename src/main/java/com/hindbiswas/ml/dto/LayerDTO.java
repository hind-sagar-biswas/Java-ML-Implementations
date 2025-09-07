package com.hindbiswas.ml.dto;

/**
 * DTO for a single layer in a neural network.
 */
public class LayerDTO extends DTO {
    /** Number of input features to the layer. */
    public int inputs;
    /** Number of perceptrons (neurons) in the layer. */
    public int perceptrons;
    /** Name of the activation function. */
    public String activationName;
    /** Weights matrix for the layer. */
    public double[][] weights;
}
