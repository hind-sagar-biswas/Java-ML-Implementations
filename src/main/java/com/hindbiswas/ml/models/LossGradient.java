package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

@FunctionalInterface
public interface LossGradient {
    SimpleMatrix apply(SimpleMatrix x, SimpleMatrix y);
}
