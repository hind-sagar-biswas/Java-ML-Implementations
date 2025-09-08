
package com.hindbiswas.ml.models;

import com.hindbiswas.ml.data.DataFrame;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Bernoulli Naive Bayes.
 */
public class BernoulliNBTest {

    private BernoulliNB nb;
    private DataFrame df;

    @BeforeEach
    void setUp() {
        nb = new BernoulliNB();

        // Toy binary dataset
        // X = features, y = labels {0, 1}
        double[][] features = {
                { 1, 0, 1 }, // class 0
                { 1, 1, 0 }, // class 0
                { 0, 1, 1 }, // class 1
                { 0, 0, 1 } // class 1
        };
        double[] labels = { 0.0, 0.0, 1.0, 1.0 };

        df = new DataFrame(3); // 3 features
        df.add(features, labels);
    }

    @Test
    @DisplayName("Predict should return correct class for training data")
    void testPredictTrainingData() {
        nb.fit(df);

        double[] sample0 = { 1, 0, 1 };
        double pred0 = nb.predict(sample0);
        assertEquals(0.0, pred0, "Expected class 0");

        double[] sample1 = { 0, 0, 1 };
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
        double[] features = { 1, 0, 1 };
        assertThrows(IllegalStateException.class, () -> nb.predict(features));
    }

    @Test
    @DisplayName("Throws if wrong feature count passed to predict")
    void testPredictWrongFeatureCount() {
        nb.fit(df);
        double[] wrongFeatures = { 1, 0 }; // only 2 features, should be 3
        assertThrows(IllegalArgumentException.class, () -> nb.predict(wrongFeatures));
    }

    @Test
    @DisplayName("Export should return true (basic smoke test)")
    void testExport() {
        nb.fit(df);
        assertDoesNotThrow(() -> {
            boolean success = nb.export(java.nio.file.Paths.get("/tmp/bernoulli_nb_model.json"));
            assertTrue(success);
        });
    }

    @Test
    @DisplayName("DTO conversion should preserve alpha and feature counts")
    void testToDTO() {
        nb.fit(df);
        var dto = nb.toDTO();
        assertEquals(nb.alpha, dto.alpha, "Alpha should match");
        assertEquals(nb.features, dto.features, "Number of features should match");
        assertEquals(nb.classes.length, dto.classes.length, "Number of classes should match");
        assertNotNull(dto.featureLogProb, "DTO featureLogProb should not be null");
    }
}
