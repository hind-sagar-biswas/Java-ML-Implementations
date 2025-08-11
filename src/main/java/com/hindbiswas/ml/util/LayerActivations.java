package com.hindbiswas.ml.util;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.models.LayerActivation;

/**
 * LayerActivations
 */
public class LayerActivations {

    public static String sigmoid() {
        return "sigmoid";
    }

    public static String softmax() {
        return "softmax";
    }

    public static String linear() {
        return "linear";
    }

    public static String tanh() {
        return "tanh";
    }

    public static String relu() {
        return "relu";
    }

    public static String leakyRelu() {
        return "leakyRelu";
    }

    public static String leakyRelu(double alpha) {
        return "leakyRelu=double:" + alpha;
    }

    public static String elu() {
        return "elu";
    }

    public static String elu(double alpha) {
        return "elu=double:" + alpha;
    }

    public static LayerActivation resolve(String name) throws IllegalArgumentException {
        String[] parts = name.split("=");
        String activationType = parts[0];
        String[] params = new String[parts.length - 1];
        System.arraycopy(parts, 1, params, 0, params.length);
        switch (activationType) {
            case "sigmoid":
                return LayerActivationFunctions.sigmoid();
            case "softmax":
                return LayerActivationFunctions.softmax();
            case "linear":
                return LayerActivationFunctions.linear();
            case "tanh":
                return LayerActivationFunctions.tanh();
            case "relu":
                return LayerActivationFunctions.relu();
            case "leakyRelu":
                if (params.length > 0 && params[0].startsWith("double:")) {
                    double alpha = Double.parseDouble(params[0].substring(7));
                    return LayerActivationFunctions.leakyRelu(alpha);
                }
                return LayerActivationFunctions.leakyRelu();
            case "elu":
                if (params.length > 0 && params[0].startsWith("double:")) {
                    double alpha = Double.parseDouble(params[0].substring(7));
                    return LayerActivationFunctions.elu(alpha);
                }
                return LayerActivationFunctions.elu();
            default:
                throw new IllegalArgumentException("Unknown activation function: " + name);
        }
    }

}

class LayerActivationFunctions {
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

            @Override
            public String toString() {
                return "sigmoid";
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

            @Override
            public String toString() {
                return "softmax";
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

            @Override
            public String toString() {
                return "linear";
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

            @Override
            public String toString() {
                return "tanh";
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

            @Override
            public String toString() {
                return "relu";
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

            @Override
            public String toString() {
                return "leakyRelu=double:" + alpha;
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

            @Override
            public String toString() {
                return "elu=double:" + alpha;
            }
        };
    }
}
