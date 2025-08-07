package com.hindbiswas.ml.models;

import org.ejml.simple.SimpleMatrix;

@FunctionalInterface
public interface LossFunction {
    SimpleMatrix apply(SimpleMatrix x, SimpleMatrix y);
}
