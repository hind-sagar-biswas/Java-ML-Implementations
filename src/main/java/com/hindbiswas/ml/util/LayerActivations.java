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

    public static LayerActivation linear() {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                return x.copy(); // identity
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                SimpleMatrix ones = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        ones.set(r, c, 1.0);
                    }
                }
                return ones;
            }
        };
    }

    public static LayerActivation tanh() {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                SimpleMatrix out = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        out.set(r, c, Math.tanh(x.get(r, c)));
                    }
                }
                return out;
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                // derivative = 1 - tanh(x)^2
                SimpleMatrix t = apply(x);
                SimpleMatrix d = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        double tv = t.get(r, c);
                        d.set(r, c, 1.0 - tv * tv);
                    }
                }
                return d;
            }
        };
    }

    public static LayerActivation relu() {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                SimpleMatrix out = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        double v = x.get(r, c);
                        out.set(r, c, v > 0 ? v : 0.0);
                    }
                }
                return out;
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                SimpleMatrix d = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        d.set(r, c, x.get(r, c) > 0 ? 1.0 : 0.0);
                    }
                }
                return d;
            }
        };
    }

    public static LayerActivation leakyRelu() {
        return leakyRelu(0.01);
    }

    public static LayerActivation leakyRelu(final double alpha) {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                SimpleMatrix out = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        double v = x.get(r, c);
                        out.set(r, c, v > 0 ? v : alpha * v);
                    }
                }
                return out;
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                SimpleMatrix d = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        d.set(r, c, x.get(r, c) > 0 ? 1.0 : alpha);
                    }
                }
                return d;
            }
        };
    }

    public static LayerActivation elu() {
        return elu(1.0);
    }

    public static LayerActivation elu(final double alpha) {
        return new LayerActivation() {
            @Override
            public SimpleMatrix apply(SimpleMatrix x) {
                SimpleMatrix out = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        double v = x.get(r, c);
                        out.set(r, c, v >= 0 ? v : alpha * (Math.exp(v) - 1.0));
                    }
                }
                return out;
            }

            @Override
            public SimpleMatrix derivative(SimpleMatrix x) {
                SimpleMatrix d = new SimpleMatrix(x.getNumRows(), x.getNumCols());
                for (int r = 0; r < x.getNumRows(); r++) {
                    for (int c = 0; c < x.getNumCols(); c++) {
                        double v = x.get(r, c);
                        d.set(r, c, v >= 0 ? 1.0 : alpha * Math.exp(v));
                    }
                }
                return d;
            }
        };
    }
}
