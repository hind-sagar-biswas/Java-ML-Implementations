package com.hindbiswas.ml.data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DataFrameTest {

    private DataFrame df3; // 3-feature DataFrame

    private static DataFrame sequentialDF(int rows, int featureCount) {
        DataFrame df = new DataFrame(featureCount, Math.max(1, rows));
        for (int r = 0; r < rows; r++) {
            double[] feats = new double[featureCount];
            for (int f = 0; f < featureCount; f++) {
                // keep values simple & distinct: row*100 + featureIndex
                feats[f] = r * 100.0 + f;
            }
            double label = r;
            df.add(feats, label);
        }
        return df;
    }

    @BeforeEach
    void setUp() {
        df3 = sequentialDF(6, 3); // 6 rows, 3 features
    }

    // ---------- Basic getters and size ----------
    @Test
    void testDimensionsAndSize() {
        assertEquals(6, df3.size());
        int[] dims = df3.shape();
        assertArrayEquals(new int[] { 6, 4 }, dims); // 3 features + 1 label
    }

    @Test
    void testGetFeaturesAndLabelsCopies() {
        double[][] feats = df3.getFeatures();
        double[] labs = df3.getLabels();
        assertEquals(6, feats.length);
        assertEquals(6, labs.length);

        // Mutating returned arrays must not change original DataFrame (getFeatures
        // returns copies)
        feats[0][0] = 9999;
        labs[0] = -1;
        double[] origRow0 = df3.getFeatures(0);
        assertNotEquals(9999, origRow0[0], 0.0);
        assertEquals(0.0, df3.getLabel(0), 0.0);
    }

    @Test
    void testGetFeaturesRefMutability() {
        double[] ref = df3.getFeaturesRef(1);
        double old = ref[0];
        ref[0] = -123.456;
        // Because getFeaturesRef returns internal reference, DataFrame should reflect
        // change
        double[] now = df3.getFeatures(1);
        assertEquals(-123.456, now[0], 1e-12);
        // restore for cleanliness
        ref[0] = old;
    }

    @Test
    void testGetAndNegativeIndexing() {
        double[] last = df3.getFeatures(-1); // last row
        assertArrayEquals(df3.getFeatures(5), last);
        Double labelNeg = df3.getLabel(-1);
        assertEquals(5.0, labelNeg.doubleValue(), 0.0);
    }

    // ---------- add variants and invalid inputs ----------
    @Test
    void testAddArrayAndArrayListVariants() {
        DataFrame df = new DataFrame(2);
        df.add(new double[] { 1.0, 2.0 }, 3.0);

        ArrayList<Double> row = new ArrayList<>();
        row.add(4.0);
        row.add(5.0);
        df.add(row, 6.0);

        assertEquals(2, df.size());
        assertArrayEquals(new double[] { 1.0, 2.0 }, df.getFeatures(0));
        assertArrayEquals(new double[] { 4.0, 5.0 }, df.getFeatures(1));
        assertEquals(6.0, df.getLabel(1), 0.0);
    }

    @Test
    void testAddMatrixWithLabels() {
        double[][] features = new double[][] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
        };
        double[] labels = new double[] { 9.0, 10.0 };
        DataFrame df = new DataFrame(3);
        df.add(features, labels);
        assertEquals(2, df.size());
        assertArrayEquals(features[1], df.getFeatures(1));
        assertEquals(10.0, df.getLabel(1), 0.0);
    }

    @Test
    void testAddArrayListOfArrayListsWithLabelIndex() {
        ArrayList<ArrayList<Double>> dataset = new ArrayList<>();
        ArrayList<Double> r0 = new ArrayList<>(Arrays.asList(1.0, 2.0, 9.0)); // last col label
        ArrayList<Double> r1 = new ArrayList<>(Arrays.asList(3.0, 4.0, 10.0));
        dataset.add(r0);
        dataset.add(r1);

        DataFrame df = new DataFrame(2);
        df.add(dataset, 2); // label index 2
        assertEquals(2, df.size());
        assertArrayEquals(new double[] { 1.0, 2.0 }, df.getFeatures(0));
        assertEquals(9.0, df.getLabel(0), 0.0);
    }

    @Test
    void testAddArrayListOfArrayListsAndLabels() {
        ArrayList<ArrayList<Double>> feats = new ArrayList<>();
        ArrayList<Double> a = new ArrayList<>(Arrays.asList(1.0, 2.0));
        ArrayList<Double> b = new ArrayList<>(Arrays.asList(3.0, 4.0));
        feats.add(a);
        feats.add(b);
        ArrayList<Double> labs = new ArrayList<>(Arrays.asList(7.0, 8.0));

        DataFrame df = new DataFrame(2);
        df.add(feats, labs);
        assertEquals(2, df.size());
        assertEquals(7.0, df.getLabel(0), 0.0);
    }

    @Test
    void testAddWithMismatchedShapesThrows() {
        // mismatched row length -> expected runtime exception (ArrayIndexOutOfBounds or
        // similar)
        double[][] features = new double[][] {
                { 1.0, 2.0 },
                { 3.0 } // short row
        };
        double[] labels = new double[] { 0.0, 1.0 };
        DataFrame df = new DataFrame(2);
        assertThrows(RuntimeException.class, () -> df.add(features, labels));
    }

    // ---------- deepCopy / clone ----------
    @Test
    void testDeepCopyIndependence() {
        DataFrame copy = df3.deepCopy();
        assertEquals(df3, copy);
        // modify original internal ref and ensure copy unaffected
        double[] ref = df3.getFeaturesRef(2);
        ref[0] = 99999.0;
        // copy should keep its original values
        assertNotEquals(df3.getFeatures(2)[0], copy.getFeatures(2)[0]);
    }

    // ---------- batch / iterateBatches ----------
    @Test
    void testBatchBasic() {
        DataFrame batch = df3.batch(1, 3);
        assertEquals(3, batch.size());
        assertArrayEquals(df3.getFeatures(1), batch.getFeatures(0));
        assertArrayEquals(df3.getFeatures(3), batch.getFeatures(2));
    }

    @Test
    void testBatchWithInvalidArgsThrows() {
        assertThrows(IllegalArgumentException.class, () -> df3.batch(0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> df3.batch(100, 1));
    }

    @Test
    void testIterateBatchesNonDivisible() {
        List<DataFrame> parts = new ArrayList<>();
        for (DataFrame b : df3.iterateBatches(4)) {
            parts.add(b);
        }
        // 6 rows with batch size 4 -> two batches (4 and 2)
        assertEquals(2, parts.size());
        assertEquals(4, parts.get(0).size());
        assertEquals(2, parts.get(1).size());
    }

    @Test
    void testIterateBatchesSeedDoesNotMutateOriginal() {
        DataFrame before = df3.deepCopy();
        // collect items from seeded iterator
        List<DataFrame> parts = new ArrayList<>();
        for (DataFrame b : df3.iterateBatches(2, 12345)) {
            parts.add(b);
        }
        // original should remain unchanged
        assertEquals(before, df3);
        // the concatenation of parts should contain same rows but shuffled order for
        // the deep copy only
        List<double[]> concatenated = new ArrayList<>();
        for (DataFrame b : parts) {
            for (int i = 0; i < b.size(); i++) {
                concatenated.add(b.getFeatures(i));
            }
        }
        // concatenated size must equal original size
        assertEquals(df3.size(), concatenated.size());
    }

    @Test
    void testIterateBatchesInvalidBatchSizeThrows() {
        assertThrows(IllegalArgumentException.class, () -> df3.iterateBatches(0));
    }

    // ---------- shuffle ----------
    @Test
    void testShuffleMutatesAndIsDeterministic() {
        DataFrame copy1 = df3.deepCopy();
        df3.shuffle(42);
        // After shuffle df3 should not equal original copy (most likely)
        assertNotEquals(copy1, df3);
        // deterministic: shuffle again with same seed on a fresh copy reproduces the
        // same result
        DataFrame fresh = copy1.deepCopy();
        fresh.shuffle(42);
        assertEquals(df3, fresh);
    }

    // ---------- iterator / DataPoint ----------
    @Test
    void testIteratorProducesDataPoints() {
        int count = 0;
        for (DataPoint dp : df3) {
            assertNotNull(dp);
            assertEquals(3, dp.features.length);
            assertEquals(count, (int) dp.label);
            count++;
        }
        assertEquals(df3.size(), count);
    }

    // ---------- remove / clear ----------
    @Test
    void testRemoveNegativeIndexRemovesLast() {
        double lastLabel = df3.getLabel(df3.size() - 1);
        double removed = df3.remove(-1);
        assertEquals(lastLabel, removed, 0.0);
        assertEquals(5, df3.size());
    }

    @Test
    void testClearResetsSize() {
        df3.clear();
        assertEquals(0, df3.size());
        // adding after clear still works
        df3.add(new double[] { 1.0, 2.0, 3.0 }, 0.0);
        assertEquals(1, df3.size());
    }

    // ---------- head / tail ----------
    @Test
    void testHeadAndTailDefault() {
        DataFrame h = df3.head();
        DataFrame t = df3.tail();
        assertEquals(5, h.size());
        assertEquals(5, t.size());
        assertArrayEquals(df3.getFeatures(0), h.getFeatures(0));
        assertArrayEquals(df3.getFeatures(df3.size() - 1), t.getFeatures(t.size() - 1));
    }

    @Test
    void testHeadTailWithLargeN() {
        DataFrame h = df3.head(100);
        assertEquals(df3.size(), h.size());
        DataFrame t = df3.tail(100);
        assertEquals(df3.size(), t.size());
    }

    @Test
    void testHeadTailWithZeroReturnsEmpty() {
        DataFrame emptyH = df3.head(0);
        DataFrame emptyT = df3.tail(0);
        assertEquals(0, emptyH.size());
        assertEquals(0, emptyT.size());
    }

    // ---------- equals / hashCode ----------
    @Test
    void testEqualsAndHashCode() {
        DataFrame a = sequentialDF(4, 2);
        DataFrame b = sequentialDF(4, 2);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());

        b.add(new double[] { 100.0, 101.0 }, 4.0);
        assertNotEquals(a, b);
    }

    // ---------- toString ----------
    @Test
    void testToStringPreviewAndMore() {
        DataFrame many = sequentialDF(12, 1); // to force "more rows" in preview
        String s = many.toString();
        assertTrue(s.contains("DataFrame(12 rows"));
        assertTrue(s.contains("... (2 more rows)"));
        // preview line present
        assertTrue(s.contains("label=0.0") || s.contains("label=0"));
    }

    // ---------- get combined row ----------
    @Test
    void testGetCombinedRow() {
        double[] combined = df3.get(2);
        assertEquals(4, combined.length); // 3 features + 1 label
        assertArrayEquals(df3.getFeatures(2), Arrays.copyOf(combined, 3));
        assertEquals(df3.getLabel(2), combined[3]);
    }

    // ---------- sanity: add DataFrame merging ----------
    @Test
    void testAddDataFrameMerges() {
        DataFrame a = sequentialDF(2, 2);
        DataFrame b = sequentialDF(3, 2);
        a.add(b);
        assertEquals(5, a.size());
        // ensure appended block matches b
        assertArrayEquals(b.getFeatures(0), a.getFeatures(2));
    }
}
