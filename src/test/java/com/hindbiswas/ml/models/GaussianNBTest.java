package com.hindbiswas.ml.models;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.hindbiswas.ml.data.DataFrame;

/**
 * Unit tests for GaussianNB using the project's DataFrame implementation.
 */
public class GaussianNBTest {

    @Test
    public void testPredictWithoutFitThrows() {
        GaussianNB gnb = new GaussianNB();
        double[] features = new double[] { 0.0 };
        assertThrows(IllegalStateException.class, () -> gnb.predict(features),
                "predict should throw when model not fitted");
    }

    @Test
    public void testFitAndPredictSimpleBinary() {
        // create DataFrame with 1 feature
        DataFrame df = new DataFrame(1);
        // class 0 around 0.0
        df.add(new double[] { 0.0 }, 0.0);
        df.add(new double[] { 0.2 }, 0.0);
        df.add(new double[] { -0.1 }, 0.0);
        // class 1 around 10.0
        df.add(new double[] { 9.8 }, 1.0);
        df.add(new double[] { 10.1 }, 1.0);
        df.add(new double[] { 10.2 }, 1.0);

        GaussianNB gnb = new GaussianNB();
        gnb.fit(df);

        Double predNearZero = gnb.predict(new double[] { 0.05 });
        Double predNearTen = gnb.predict(new double[] { 10.0 });

        assertNotNull(predNearZero);
        assertNotNull(predNearTen);

        // Compare numeric values (labels in DataFrame are doubles)
        assertEquals(0.0, predNearZero.doubleValue(), 1e-8, "Prediction near 0 should be class 0.0");
        assertEquals(1.0, predNearTen.doubleValue(), 1e-8, "Prediction near 10 should be class 1.0");
    }

    @Test
    public void testPredictWithDifferentFeatureLengthThrows() {
        DataFrame df = new DataFrame(2);
        df.add(new double[] { 1.0, 2.0 }, 0.0);
        GaussianNB gnb = new GaussianNB();
        gnb.fit(df);

        double[] wrong = new double[] { 1.0 }; // wrong length
        assertThrows(IllegalArgumentException.class, () -> gnb.predict(wrong));
    }
}
