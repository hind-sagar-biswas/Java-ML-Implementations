package com.hindbiswas.ml.data;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Random;
import java.util.RandomAccess;

/**
 * Data
 */
public class DataFrame implements Cloneable, RandomAccess, Iterable<DataPoint> {
    private static final int DEFAULT_CAPACITY = 10;
    private static final double[] EMPTY_LABELS_DATA = {};

    private double[][] featureElementData;
    private double[] labelElementData;

    private final int featureCount;
    private int capacity;
    private int length;

    public DataFrame(int featureCount) throws IllegalArgumentException {
        if (featureCount <= 0) {
            throw new IllegalArgumentException("Feature count must be greater than 0.");
        }

        this.featureCount = featureCount;
        this.capacity = DEFAULT_CAPACITY;
    }

    public DataFrame(int featureCount, int capacity) throws IllegalArgumentException {
        if (featureCount <= 0) {
            throw new IllegalArgumentException("Feature count must be greater than 0.");
        }
        if (capacity < 1) {
            throw new IllegalArgumentException("Capacity must be greater than 0.");
        }

        this.featureCount = featureCount;
        this.capacity = capacity;
    }

    public DataFrame add(double[] features, double label) throws IllegalArgumentException {
        if (features.length != featureCount) {
            throw new IllegalArgumentException("Feature count does not match the DataFrame's feature count.");
        }

        if (featureElementData == null) {
            featureElementData = new double[capacity][featureCount];
            labelElementData = new double[capacity];
        }

        if (length == capacity) {
            grow();
        }

        for (int i = 0; i < featureCount; i++) {
            featureElementData[length][i] = features[i];
        }
        labelElementData[length] = label;
        length++;

        return this;
    }

    public DataFrame add(ArrayList<Double> features, double label)
            throws IllegalArgumentException, NullPointerException {
        Double[] featuresArray = features.toArray(new Double[0]);
        double[] featuresDouble = new double[featuresArray.length];
        for (int i = 0; i < featuresArray.length; i++) {
            if (featuresArray[i] == null) {
                throw new NullPointerException("Features cannot contain null values.");
            }
            featuresDouble[i] = featuresArray[i];
        }
        return add(featuresDouble, label);
    }

    public DataFrame add(double[][] features, double[] labels) throws IllegalArgumentException {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("Features and labels cannot be null.");
        }
        if (features.length == 0 || labels.length == 0) {
            throw new IllegalArgumentException("Features and labels cannot be empty.");
        }
        if (features.length != labels.length) {
            throw new IllegalArgumentException("Feature and label arrays must have the same length.");
        }
        if (features[0].length != featureCount) {
            throw new IllegalArgumentException("Feature count does not match the DataFrame's feature count.");
        }

        adjust(features.length);

        for (int i = 0; i < features.length; i++) {
            for (int j = 0; j < featureCount; j++) {
                featureElementData[length + i][j] = features[i][j];
            }
            labelElementData[length + i] = labels[i];
        }
        length += features.length;

        return this;
    }

    public DataFrame add(ArrayList<ArrayList<Double>> features, ArrayList<Double> labels)
            throws IllegalArgumentException, NullPointerException {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("Features and labels cannot be null.");
        }

        if (features.size() == 0 || labels.size() == 0) {
            throw new IllegalArgumentException("Features and labels cannot be empty.");
        }

        adjust(features.size());

        for (int i = 0; i < features.size(); i++) {
            if (labels.get(i) == null) {
                throw new NullPointerException("Labels cannot contain null values.");
            }
            add(features.get(i), labels.get(i));
        }

        return this;
    }

    public DataFrame add(double[][] dataset, int labelIndex) throws IllegalArgumentException {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
        if (dataset.length == 0) {
            throw new IllegalArgumentException("Dataset cannot be empty.");
        }
        if (labelIndex < 0 || labelIndex >= dataset[0].length) {
            throw new IllegalArgumentException("Label index is out of bounds.");
        }
        if (dataset[0].length - 1 != featureCount) {
            throw new IllegalArgumentException("Feature count does not match the DataFrame's feature count.");
        }

        adjust(dataset.length);

        for (int i = 0; i < dataset.length; i++) {
            int k = 0;
            for (int j = 0; j < dataset[i].length; j++) {
                if (j == labelIndex)
                    continue;
                featureElementData[length + i][k++] = dataset[i][j];
            }
            labelElementData[length + i] = dataset[i][labelIndex];
        }
        length += dataset.length;
        return this;
    }

    public DataFrame add(ArrayList<ArrayList<Double>> dataset, int labelIndex)
            throws IllegalArgumentException, IndexOutOfBoundsException, NullPointerException {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
        if (dataset.size() == 0) {
            throw new IllegalArgumentException("Dataset cannot be empty.");
        }
        int cols = dataset.get(0).size();
        if (labelIndex < 0 || labelIndex >= cols) {
            throw new IndexOutOfBoundsException("Label index is out of bounds.");
        }
        if (cols - 1 != featureCount) {
            throw new IllegalArgumentException("Feature count does not match the DataFrame's feature count.");
        }

        adjust(dataset.size());

        for (int i = 0; i < dataset.size(); i++) {
            ArrayList<Double> row = dataset.get(i);
            if (row == null) {
                throw new NullPointerException("Row cannot be null.");
            }
            if (row.size() != cols) {
                throw new IllegalArgumentException("All rows must have the same number of columns.");
            }
            int k = 0;
            for (int j = 0; j < row.size(); j++) {
                Double val = row.get(j);
                if (val == null) {
                    throw new NullPointerException("Null value found in dataset.");
                }
                if (j == labelIndex) {
                    labelElementData[length + i] = val;
                } else {
                    featureElementData[length + i][k++] = val;
                }
            }
        }
        length += dataset.size();
        return this;
    }

    public DataFrame add(DataFrame df) throws IllegalArgumentException {
        if (df == null) {
            throw new IllegalArgumentException("DataFrame cannot be null.");
        }

        int[] dimensions = df.dimensions();
        if (dimensions[1] - 1 != featureCount) {
            throw new IllegalArgumentException("Feature count does not match the DataFrame's feature count.");
        }

        adjust(dimensions[0]);

        double[][] features = df.getFeatures();
        double[] labels = df.getLabels();

        for (int i = 0; i < dimensions[0]; i++) {
            for (int j = 0; j < featureCount; j++) {
                featureElementData[length + i][j] = features[i][j];
            }
            labelElementData[length + i] = labels[i];
        }
        length += dimensions[0];

        return this;
    }

    public double remove(int index) {
        if (index < 0) {
            index = length + index;
        }
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException();
        }

        double removedLabel = labelElementData[index];
        if (index < length - 1) {
            System.arraycopy(labelElementData, index + 1, labelElementData, index, length - index - 1);
            for (int i = index; i < length - 1; i++) {
                System.arraycopy(featureElementData[i + 1], 0, featureElementData[i], 0, featureCount);
            }
        }
        for (int j = 0; j < featureCount; j++) {
            featureElementData[length - 1][j] = 0.0;
        }
        labelElementData[length - 1] = 0.0;
        length--;
        return removedLabel;
    }

    public void clear() {
        if (featureElementData == null || labelElementData == null) {
            length = 0;
            return;
        }
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < featureCount; j++) {
                featureElementData[i][j] = 0.0;
            }
            labelElementData[i] = 0.0;
        }
        length = 0;
    }

    public int size() {
        return length;
    }

    public int[] dimensions() {
        return new int[] { length, featureCount + 1 };
    }

    public double[][] getFeatures() {
        if (featureElementData == null)
            return new double[0][0];
        double[][] out = new double[length][featureCount];
        for (int i = 0; i < length; i++) {
            System.arraycopy(featureElementData[i], 0, out[i], 0, featureCount);
        }
        return out;
    }

    public double[] getLabels() {
        if (labelElementData == null) {
            return DataFrame.EMPTY_LABELS_DATA;
        }
        double[] out = new double[length];
        System.arraycopy(labelElementData, 0, out, 0, length);
        return out;
    }

    public double[] getFeatures(int index) {
        if (index < 0) {
            index = length + index;
        }

        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException();
        }
        double[] out = new double[featureCount];
        System.arraycopy(featureElementData[index], 0, out, 0, featureCount);
        return out;
    }

    public double[] getFeaturesRef(int index) {
        if (index < 0) {
            index = length + index;
        }

        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException();
        }
        return featureElementData[index];
    }

    public Double getLabel(int index) {
        if (index < 0) {
            index = length + index;
        }

        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException();
        }
        return labelElementData[index];
    }

    public double[] get(int index) {
        double[] data = new double[featureCount + 1];
        double[] features = getFeatures(index);
        data[featureCount] = labelElementData[index];
        for (int i = 0; i < featureCount; i++) {
            data[i] = features[i];
        }
        return data;
    }

    public DataFrame deepCopy() {
        DataFrame df = new DataFrame(featureCount, capacity);
        if (this.length == 0) {
            df.featureElementData = new double[0][featureCount];
            df.labelElementData = new double[0];
            df.capacity = 0;
            return df;
        }
        df.featureElementData = new double[this.length][featureCount];
        df.labelElementData = new double[this.length];

        for (int i = 0; i < this.length; i++) {
            System.arraycopy(this.featureElementData[i], 0, df.featureElementData[i], 0, featureCount);
        }
        System.arraycopy(this.labelElementData, 0, df.labelElementData, 0, this.length);
        df.length = this.length;
        df.capacity = this.length;
        return df;
    }

    @Override
    public DataFrame clone() {
        return deepCopy();
    }

    public DataFrame batch(int start, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize must be greater than 0");
        if (start < 0 || start >= length)
            throw new IndexOutOfBoundsException();
        int end = Math.min(start + batchSize, length);

        DataFrame batch = new DataFrame(featureCount, end - start);
        batch.featureElementData = new double[end - start][featureCount];
        batch.labelElementData = new double[end - start];

        for (int i = 0; i < end - start; i++) {
            System.arraycopy(featureElementData[start + i], 0, batch.featureElementData[i], 0, featureCount);
            batch.labelElementData[i] = labelElementData[start + i];
        }
        batch.length = end - start;
        return batch;
    }

    public void shuffle(int seed) {
        if (length <= 1) {
            return;
        }

        Random rng = new Random(seed);
        for (int i = length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);

            double[] tempF = featureElementData[i];
            featureElementData[i] = featureElementData[j];
            featureElementData[j] = tempF;

            double tempL = labelElementData[i];
            labelElementData[i] = labelElementData[j];
            labelElementData[j] = tempL;
        }
    }

    public Iterable<DataFrame> iterateBatches(int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize must be greater than 0");
        return () -> new Iterator<DataFrame>() {
            private int cursor = 0;

            @Override
            public boolean hasNext() {
                return cursor < length;
            }

            @Override
            public DataFrame next() {
                if (!hasNext())
                    throw new NoSuchElementException();
                int start = cursor;
                int end = Math.min(start + batchSize, length);
                DataFrame batch = batch(start, end - start);
                cursor += (end - start);
                return batch;
            }
        };
    }

    public Iterable<DataFrame> iterateBatches(int batchSize, int seed) {
        DataFrame copy = this.deepCopy();
        copy.shuffle(seed);
        return () -> new Iterator<DataFrame>() {
            private int cursor = 0;

            @Override
            public boolean hasNext() {
                return cursor < copy.length;
            }

            @Override
            public DataFrame next() {
                if (!hasNext())
                    throw new NoSuchElementException();
                int start = cursor;
                int end = Math.min(start + batchSize, copy.length);
                DataFrame batch = copy.batch(start, end - start);
                cursor += (end - start);
                return batch;
            }
        };
    }

    @Override
    public Iterator<DataPoint> iterator() {
        return new Iterator<DataPoint>() {
            private int index = 0;

            @Override
            public boolean hasNext() {
                return index < length;
            }

            @Override
            public DataPoint next() {
                if (!hasNext())
                    throw new NoSuchElementException();
                double[] features = getFeatures(index);
                double label = getLabel(index);
                DataPoint dp = new DataPoint(features, label);
                index++;
                return dp;
            }
        };
    }

    public void trimToSize() {
        if (featureElementData == null)
            return;
        if (capacity == length)
            return;

        double[][] newFeatures = new double[length][featureCount];
        double[] newLabels = new double[length];
        for (int i = 0; i < length; i++) {
            System.arraycopy(featureElementData[i], 0, newFeatures[i], 0, featureCount);
        }
        System.arraycopy(labelElementData, 0, newLabels, 0, length);
        featureElementData = newFeatures;
        labelElementData = newLabels;
        capacity = length;
    }

    @Override
    public String toString() {
        int rows = this.length;
        int cols = this.featureCount + 1;
        StringBuilder sb = new StringBuilder();
        sb.append("DataFrame(").append(rows).append(" rows, ").append(cols).append(" cols)\n");

        int preview = Math.min(rows, 10);
        double[][] feats = getFeatures();
        double[] labs = getLabels();

        for (int i = 0; i < preview; i++) {
            sb.append("[");
            for (int j = 0; j < featureCount; j++) {
                sb.append(feats[i][j]);
                if (j < featureCount - 1)
                    sb.append(", ");
            }
            sb.append(", label=").append(labs[i]).append("]\n");
        }
        if (rows > preview) {
            sb.append("... (").append(rows - preview).append(" more rows)\n");
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof DataFrame))
            return false;
        DataFrame that = (DataFrame) o;
        if (this.featureCount != that.featureCount)
            return false;
        if (this.length != that.length)
            return false;

        return Arrays.deepEquals(this.getFeatures(), that.getFeatures())
                && Arrays.equals(this.getLabels(), that.getLabels());
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(featureCount, length);
        result = 31 * result + Arrays.deepHashCode(this.getFeatures());
        result = 31 * result + Arrays.hashCode(this.getLabels());
        return result;
    }

    public DataFrame head() {
        return head(5);
    }

    public DataFrame head(int n) {
        if (n <= 0) {
            DataFrame empty = new DataFrame(featureCount, 1);
            empty.featureElementData = new double[0][featureCount];
            empty.labelElementData = new double[0];
            empty.length = 0;
            empty.capacity = 0;
            return empty;
        }
        int actual = Math.min(n, this.length);
        DataFrame out = new DataFrame(featureCount, Math.max(1, actual));
        out.featureElementData = new double[actual][featureCount];
        out.labelElementData = new double[actual];
        for (int i = 0; i < actual; i++) {
            System.arraycopy(this.featureElementData[i], 0, out.featureElementData[i], 0, featureCount);
            out.labelElementData[i] = this.labelElementData[i];
        }
        out.length = actual;
        out.capacity = actual;
        return out;
    }

    public DataFrame tail() {
        return tail(5);
    }

    public DataFrame tail(int n) {
        if (n <= 0) {
            DataFrame empty = new DataFrame(featureCount, 1);
            empty.featureElementData = new double[0][featureCount];
            empty.labelElementData = new double[0];
            empty.length = 0;
            empty.capacity = 0;
            return empty;
        }
        int actual = Math.min(n, this.length);
        int start = this.length - actual;
        DataFrame out = new DataFrame(featureCount, Math.max(1, actual));
        out.featureElementData = new double[actual][featureCount];
        out.labelElementData = new double[actual];
        for (int i = 0; i < actual; i++) {
            System.arraycopy(this.featureElementData[start + i], 0, out.featureElementData[i], 0, featureCount);
            out.labelElementData[i] = this.labelElementData[start + i];
        }
        out.length = actual;
        out.capacity = actual;
        return out;
    }

    public String summary() {
        if (this.length == 0) {
            return "Empty DataFrame (0 rows).\n";
        }

        int cols = featureCount + 1;
        String[] colNames = new String[cols];
        for (int c = 0; c < featureCount; c++) {
            colNames[c] = "feature" + c;
        }
        colNames[featureCount] = "label";

        String[] statsNames = new String[] { "count", "mean", "std", "min", "25%", "50%", "75%", "max" };
        double[][] stats = new double[statsNames.length][cols];

        for (int c = 0; c < cols; c++) {
            double[] colData = (c < featureCount) ? getColumnValues(c) : getLabels();

            stats[0][c] = colData.length;
            stats[1][c] = mean(colData);
            stats[2][c] = stdSample(colData);
            stats[3][c] = min(colData);
            stats[4][c] = percentile(colData, 0.25);
            stats[5][c] = percentile(colData, 0.50);
            stats[6][c] = percentile(colData, 0.75);
            stats[7][c] = max(colData);
        }

        int nameColWidth = 8;
        int colWidth = 14;
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("%-" + nameColWidth + "s", ""));
        for (int c = 0; c < cols; c++) {
            sb.append(String.format("%" + colWidth + "s", colNames[c]));
        }
        sb.append("\n");

        DecimalFormat df = new DecimalFormat("#.######");

        for (int r = 0; r < statsNames.length; r++) {
            sb.append(String.format("%-" + nameColWidth + "s", statsNames[r]));
            for (int c = 0; c < cols; c++) {
                if ("count".equals(statsNames[r])) {

                    sb.append(String.format("%" + colWidth + "d", (int) stats[r][c]));
                } else {
                    double v = stats[r][c];
                    String s;
                    if (Double.isNaN(v)) {
                        s = "NaN";
                    } else {
                        s = formatDouble(df, v);
                    }
                    sb.append(String.format("%" + colWidth + "s", s));
                }
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    private void adjust(int amount) {
        if (featureElementData == null) {
            if (amount > capacity) {
                capacity = amount + 2;
            }

            featureElementData = new double[capacity][featureCount];
            labelElementData = new double[capacity];
        } else if (length + amount > capacity) {
            grow(length + amount + 2);
        }
    }

    private void grow() {
        int newCapacity = Math.max(capacity + (capacity >> 1), capacity + 1);
        grow(newCapacity);
    }

    private void grow(int minRequiredCapacity) {
        int newCapacity = Math.max(minRequiredCapacity, Math.max(capacity + (capacity >> 1), capacity + 1));
        if (newCapacity < 0) {
            newCapacity = Integer.MAX_VALUE;
        }

        double[][] newFeatures = new double[newCapacity][featureCount];
        double[] newLabels = new double[newCapacity];

        for (int i = 0; i < length; i++) {
            System.arraycopy(featureElementData[i], 0, newFeatures[i], 0, featureCount);
        }
        System.arraycopy(labelElementData, 0, newLabels, 0, length);

        featureElementData = newFeatures;
        labelElementData = newLabels;
        capacity = newCapacity;
    }

    private double[] getColumnValues(int col) {
        if (featureElementData == null || length == 0) {
            return new double[0];
        }
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = featureElementData[i][col];
        }
        return out;
    }

    private double percentile(double[] arr, double p) {
        if (arr == null || arr.length == 0)
            return Double.NaN;
        double[] copy = Arrays.copyOf(arr, arr.length);
        Arrays.sort(copy);
        if (copy.length == 1)
            return copy[0];
        double pos = p * (copy.length - 1);
        int lower = (int) Math.floor(pos);
        int upper = (int) Math.ceil(pos);
        if (lower == upper)
            return copy[lower];
        double lowerVal = copy[lower];
        double upperVal = copy[upper];
        double frac = pos - lower;
        return lowerVal + frac * (upperVal - lowerVal);
    }

    private double mean(double[] arr) {
        if (arr == null || arr.length == 0)
            return Double.NaN;
        double s = 0.0;
        for (double v : arr)
            s += v;
        return s / arr.length;
    }

    private double stdSample(double[] arr) {
        if (arr == null || arr.length == 0)
            return Double.NaN;
        int n = arr.length;
        if (n < 2)
            return Double.NaN;
        double mu = mean(arr);
        double s2 = 0.0;
        for (double v : arr) {
            double d = v - mu;
            s2 += d * d;
        }
        return Math.sqrt(s2 / (n - 1));
    }

    private double min(double[] arr) {
        if (arr == null || arr.length == 0)
            return Double.NaN;
        double m = Double.POSITIVE_INFINITY;
        for (double v : arr)
            if (v < m)
                m = v;
        return m;
    }

    private double max(double[] arr) {
        if (arr == null || arr.length == 0)
            return Double.NaN;
        double m = Double.NEGATIVE_INFINITY;
        for (double v : arr)
            if (v > m)
                m = v;
        return m;
    }

    private String formatDouble(DecimalFormat df, double v) {
        if (Double.isNaN(v))
            return "NaN";
        if (v == Math.rint(v) && Math.abs(v) < 1e12) {
            return String.format("%.0f", v);
        }
        return df.format(v);
    }

}
