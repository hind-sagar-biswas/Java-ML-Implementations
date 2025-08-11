package com.hindbiswas.ml.util;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.models.LayerActivation;

/**
 * LayerActivations
 */
public class LayerActivations {
    public static LayerActivation sigmoid() {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                return x.negative().elementExp().plus(1).elementPower(-1);
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                SimpleMatrix sigmoid = apply(x);
                return sigmoid.elementMult(sigmoid.minus(1).negative());
            }
        };
    }

    public static LayerActivation softmax() {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                double max = x.elementMax();
                SimpleMatrix stabilized = x.minus(max);
                SimpleMatrix exp = stabilized.elementExp();
                double sum = exp.elementSum();

                return exp.divide(sum);
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                throw new UnsupportedOperationException(
                        "Softmax derivative (Jacobian) is not supported for elementwise backprop. Use softmax only as final layer with cross-entropy.");
            }
        };
    }
}
