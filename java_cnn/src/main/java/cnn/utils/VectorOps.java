package cnn.utils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized tensor operations using Java Vector API
 */
public class VectorOps {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    /**
     * Element-wise addition of two arrays using Vector API
     */
    public static float[] add(float[] a, float[] b) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        
        return result;
    }
    
    /**
     * Element-wise subtraction using Vector API
     */
    public static float[] subtract(float[] a, float[] b) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.sub(vb).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        
        return result;
    }
    
    /**
     * Element-wise multiplication using Vector API
     */
    public static float[] multiply(float[] a, float[] b) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.mul(vb).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        
        return result;
    }
    
    /**
     * Scalar multiplication using Vector API
     */
    public static float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            va.mul(scalar).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }
        
        return result;
    }
    
    /**
     * Dot product using Vector API
     */
    public static float dot(float[] a, float[] b) {
        float sum = 0;
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        FloatVector sumVector = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            sumVector = sumVector.add(va.mul(vb));
        }
        
        sum = sumVector.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    /**
     * Sum of array elements using Vector API
     */
    public static float sum(float[] a) {
        float sum = 0;
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        FloatVector sumVector = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            sumVector = sumVector.add(va);
        }
        
        sum = sumVector.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        
        for (; i < a.length; i++) {
            sum += a[i];
        }
        
        return sum;
    }
    
    /**
     * Max of array elements using Vector API
     */
    public static float max(float[] a) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        if (a.length >= SPECIES.length()) {
            FloatVector maxVector = FloatVector.fromArray(SPECIES, a, 0);
            for (i = SPECIES.length(); i < upperBound; i += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, a, i);
                maxVector = maxVector.max(va);
            }
            maxVal = maxVector.reduceLanes(jdk.incubator.vector.VectorOperators.MAX);
        }
        
        for (; i < a.length; i++) {
            if (a[i] > maxVal) maxVal = a[i];
        }
        
        return maxVal;
    }
    
    /**
     * ReLU activation using Vector API
     */
    public static float[] relu(float[] a) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            va.max(zero).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = Math.max(0, a[i]);
        }
        
        return result;
    }
    
    /**
     * ReLU derivative: 1 if x > 0, 0 otherwise
     */
    public static float[] reluDerivative(float[] a) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] > 0 ? 1.0f : 0.0f;
        }
        return result;
    }
    
    /**
     * Softmax activation
     */
    public static float[] softmax(float[] a) {
        float[] result = new float[a.length];
        float maxVal = max(a);
        
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            result[i] = (float) Math.exp(a[i] - maxVal);
            sum += result[i];
        }
        
        for (int i = 0; i < a.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    /**
     * Matrix-vector multiplication using Vector API
     */
    public static float[] matVecMul(float[][] matrix, float[] vec) {
        float[] result = new float[matrix.length];
        
        for (int i = 0; i < matrix.length; i++) {
            result[i] = dot(matrix[i], vec);
        }
        
        return result;
    }
    
    /**
     * Transpose matrix
     */
    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] result = new float[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
}
