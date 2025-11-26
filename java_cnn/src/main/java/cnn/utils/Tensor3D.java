package cnn.utils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * 3D Tensor representation for CNN operations with Java Vector API optimization
 * Dimensions: [channels][height][width]
 * Uses SIMD vectorization for accelerated operations
 */
public class Tensor3D {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    public float[][][] data;
    public int channels;
    public int height;
    public int width;
    
    public Tensor3D(int channels, int height, int width) {
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.data = new float[channels][height][width];
    }
    
    public Tensor3D(float[][][] data) {
        this.data = data;
        this.channels = data.length;
        this.height = data[0].length;
        this.width = data[0][0].length;
    }
    
    /**
     * Create tensor from 2D array (single channel)
     */
    public static Tensor3D fromArray2D(float[][] arr) {
        Tensor3D tensor = new Tensor3D(1, arr.length, arr[0].length);
        tensor.data[0] = arr;
        return tensor;
    }
    
    /**
     * Flatten tensor to 1D array using vectorized copy
     */
    public float[] flatten() {
        float[] result = new float[channels * height * width];
        int idx = 0;
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                // Vectorized copy for each row
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, data[c][h], w);
                    v.intoArray(result, idx + w);
                }
                
                // Handle remaining elements
                for (; w < width; w++) {
                    result[idx + w] = data[c][h][w];
                }
                
                idx += width;
            }
        }
        return result;
    }
    
    /**
     * Reshape 1D array to tensor using vectorized copy
     */
    public static Tensor3D fromFlattened(float[] flat, int channels, int height, int width) {
        Tensor3D tensor = new Tensor3D(channels, height, width);
        int idx = 0;
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                // Vectorized copy for each row
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, flat, idx + w);
                    v.intoArray(tensor.data[c][h], w);
                }
                
                // Handle remaining elements
                for (; w < width; w++) {
                    tensor.data[c][h][w] = flat[idx + w];
                }
                
                idx += width;
            }
        }
        return tensor;
    }
    
    /**
     * Get total number of elements
     */
    public int size() {
        return channels * height * width;
    }
    
    /**
     * Apply ReLU activation using vectorized operations
     */
    public Tensor3D relu() {
        Tensor3D result = new Tensor3D(channels, height, width);
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                // Vectorized ReLU
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, data[c][h], w);
                    v.max(zero).intoArray(result.data[c][h], w);
                }
                
                // Handle remaining elements
                for (; w < width; w++) {
                    result.data[c][h][w] = Math.max(0, data[c][h][w]);
                }
            }
        }
        return result;
    }
    
    /**
     * Create a copy using vectorized operations
     */
    public Tensor3D copy() {
        Tensor3D result = new Tensor3D(channels, height, width);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                // Vectorized copy
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, data[c][h], w);
                    v.intoArray(result.data[c][h], w);
                }
                
                // Handle remaining elements using arraycopy for efficiency
                if (w < width) {
                    System.arraycopy(data[c][h], w, result.data[c][h], w, width - w);
                }
            }
        }
        return result;
    }
    
    /**
     * Zero padding with vectorized fill
     */
    public Tensor3D pad(int padding) {
        if (padding == 0) return this;
        
        int newHeight = height + 2 * padding;
        int newWidth = width + 2 * padding;
        Tensor3D result = new Tensor3D(channels, newHeight, newWidth);
        
        // Copy data to padded tensor
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int destH = h + padding;
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                // Vectorized copy for the row data
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, data[c][h], w);
                    v.intoArray(result.data[c][destH], w + padding);
                }
                
                // Handle remaining elements
                for (; w < width; w++) {
                    result.data[c][destH][w + padding] = data[c][h][w];
                }
            }
        }
        return result;
    }
    
    /**
     * Element-wise addition with another tensor using vectorized operations
     */
    public Tensor3D add(Tensor3D other) {
        Tensor3D result = new Tensor3D(channels, height, width);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v1 = FloatVector.fromArray(SPECIES, data[c][h], w);
                    FloatVector v2 = FloatVector.fromArray(SPECIES, other.data[c][h], w);
                    v1.add(v2).intoArray(result.data[c][h], w);
                }
                
                for (; w < width; w++) {
                    result.data[c][h][w] = data[c][h][w] + other.data[c][h][w];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Scale tensor by a scalar using vectorized operations
     */
    public Tensor3D scale(float scalar) {
        Tensor3D result = new Tensor3D(channels, height, width);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, data[c][h], w);
                    v.mul(scalar).intoArray(result.data[c][h], w);
                }
                
                for (; w < width; w++) {
                    result.data[c][h][w] = data[c][h][w] * scalar;
                }
            }
        }
        
        return result;
    }
}
