package cnn.utils;

/**
 * 3D Tensor representation for CNN operations
 * Dimensions: [channels][height][width]
 */
public class Tensor3D {
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
     * Flatten tensor to 1D array
     */
    public float[] flatten() {
        float[] result = new float[channels * height * width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[idx++] = data[c][h][w];
                }
            }
        }
        return result;
    }
    
    /**
     * Reshape 1D array to tensor
     */
    public static Tensor3D fromFlattened(float[] flat, int channels, int height, int width) {
        Tensor3D tensor = new Tensor3D(channels, height, width);
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    tensor.data[c][h][w] = flat[idx++];
                }
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
     * Apply ReLU activation in place
     */
    public Tensor3D relu() {
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result.data[c][h][w] = Math.max(0, data[c][h][w]);
                }
            }
        }
        return result;
    }
    
    /**
     * Create a copy
     */
    public Tensor3D copy() {
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(data[c][h], 0, result.data[c][h], 0, width);
            }
        }
        return result;
    }
    
    /**
     * Zero padding
     */
    public Tensor3D pad(int padding) {
        if (padding == 0) return this;
        
        Tensor3D result = new Tensor3D(channels, height + 2 * padding, width + 2 * padding);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result.data[c][h + padding][w + padding] = data[c][h][w];
                }
            }
        }
        return result;
    }
}
