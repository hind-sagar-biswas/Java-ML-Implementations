package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

public interface LayerActivation {
    SimpleMatrix apply(SimpleMatrix x);

    SimpleMatrix derivative(SimpleMatrix x);
}
