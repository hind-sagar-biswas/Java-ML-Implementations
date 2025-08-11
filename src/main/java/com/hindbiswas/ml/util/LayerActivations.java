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
                SimpleMatrix s = apply(x);
                int n = s.getNumRows();
                SimpleMatrix jacobian = new SimpleMatrix(n, n);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (i == j) {
                            jacobian.set(i, j, s.get(i) * (1 - s.get(i)));
                        } else {
                            jacobian.set(i, j, -s.get(i) * s.get(j));
                        }
                    }
                }
                return jacobian;
            }
        };
    }
}
