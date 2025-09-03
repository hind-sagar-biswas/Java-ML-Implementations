
package com.hindbiswas.ml.models;

import com.hindbiswas.ml.data.DataFrame;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Multinomial Naive Bayes.
 */
public class MultinomialNBTest {

    private MultinomialNB nb;
    private DataFrame df;

    @BeforeEach
    void setUp() {
        nb = new MultinomialNB();

        // Simple dataset (like word counts):
        // X = features, y = labels {0, 1}
        double[][] features = {
                { 2, 1 }, // class 0
                { 1, 1 }, // class 0
                { 1, 2 }, // class 1
                { 0, 3 } // class 1
        };
        double[] labels = { 0.0, 0.0, 1.0, 1.0 };

        df = new DataFrame(2);
        df.add(features, labels);
    }

    @Test
    @DisplayName("Fit should set fitted=true and compute priors")
    void testFitSetsPriors() {
        nb.fit(df);
        assertTrue(nb.fitted, "Model should be marked as fitted");
        assertNotNull(nb.logClassPriors, "Class priors should be set");
    }

    @Test
    @DisplayName("Predict should return correct class for training data")
    void testPredictTrainingData() {
        nb.fit(df);

        double[] sample0 = { 2, 1 };
        double pred0 = nb.predict(sample0);
        assertEquals(0.0, pred0, "Expected class 0");

        double[] sample1 = { 0, 3 };
        double pred1 = nb.predict(sample1);
        assertEquals(1.0, pred1, "Expected class 1");
    }

    @Test
    @DisplayName("Score should return 1.0 on training data")
    void testScoreOnTrainingData() {
        nb.fit(df);
        double acc = nb.score(df);
        assertEquals(1.0, acc, 1e-9, "Model should perfectly classify training set");
    }

    @Test
    @DisplayName("Throws if predict called before fit")
    void testPredictBeforeFit() {
        double[] features = { 1, 0 };
        assertThrows(IllegalStateException.class, () -> nb.predict(features));
    }

    @Test
    @DisplayName("Throws if wrong feature count passed to predict")
    void testPredictWrongFeatureCount() {
        nb.fit(df);
        double[] wrongFeatures = { 1.0 }; // only 1 feature, should be 2
        assertThrows(IllegalArgumentException.class, () -> nb.predict(wrongFeatures));
    }

    @Test
    @DisplayName("Export should return true (basic smoke test)")
    void testExport() {
        nb.fit(df);
        assertDoesNotThrow(() -> {
            boolean success = nb.export(java.nio.file.Paths.get("/tmp/nb_model.json"));
            assertTrue(success);
        });
    }
}
