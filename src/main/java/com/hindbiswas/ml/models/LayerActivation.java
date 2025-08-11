package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

public interface LayerActivation {
    public SimpleMatrix apply(SimpleMatrix x);

    public SimpleMatrix derivative(SimpleMatrix x);

    public String toString();
}
