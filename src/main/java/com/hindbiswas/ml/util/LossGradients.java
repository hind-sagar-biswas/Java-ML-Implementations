package com.hindbiswas.ml.util;

import com.hindbiswas.ml.models.LossGradient;

/**
 * LossGradients
 */
public class LossGradients {

    public static String softmaxCrossEntropy() {
        return "softmaxCrossEntropy";
    }

    public static LossGradient resolve(String name) throws IllegalArgumentException {
        switch (name) {
            case "softmaxCrossEntropy":
                return LossGradientsFunctions.softmaxCrossEntropy();
            default:
                throw new IllegalArgumentException("Unknown loss gradient: " + name);
        }
    }
}

class LossGradientsFunctions {
    public static LossGradient softmaxCrossEntropy() {
        return (pred, label) -> {
            if (pred.getNumRows() != label.getNumRows() || pred.getNumCols() != label.getNumCols()) {
                throw new IllegalArgumentException("Input and output matrices must have the same dimensions.");
            }

            return pred.minus(label);
        };
    }
}
