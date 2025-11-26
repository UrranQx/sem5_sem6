package cnn.benchmark;

/**
 * Scalar (non-vectorized) operations for comparison with VectorOps
 * These implementations use standard Java loops without SIMD optimization
 */
public class ScalarOps {
    
    /**
     * Element-wise addition using scalar operations
     */
    public static float[] add(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    /**
     * Element-wise subtraction using scalar operations
     */
    public static float[] subtract(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }
    
    /**
     * Element-wise multiplication using scalar operations
     */
    public static float[] multiply(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    /**
     * Scalar multiplication using scalar operations
     */
    public static float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }
        return result;
    }
    
    /**
     * Dot product using scalar operations
     */
    public static float dot(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * Sum of array elements using scalar operations
     */
    public static float sum(float[] a) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i];
        }
        return sum;
    }
    
    /**
     * Max of array elements using scalar operations
     */
    public static float max(float[] a) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > maxVal) {
                maxVal = a[i];
            }
        }
        return maxVal;
    }
    
    /**
     * ReLU activation using scalar operations
     */
    public static float[] relu(float[] a) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = Math.max(0, a[i]);
        }
        return result;
    }
    
    /**
     * ReLU derivative using scalar operations
     */
    public static float[] reluDerivative(float[] a) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] > 0 ? 1.0f : 0.0f;
        }
        return result;
    }
    
    /**
     * Softmax activation using scalar operations
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
     * Matrix-vector multiplication using scalar operations
     */
    public static float[] matVecMul(float[][] matrix, float[] vec) {
        float[] result = new float[matrix.length];
        
        for (int i = 0; i < matrix.length; i++) {
            result[i] = dot(matrix[i], vec);
        }
        
        return result;
    }
    
    /**
     * 3D tensor ReLU using scalar operations
     */
    public static void relu3D(float[][][] input, float[][][] output) {
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[c][h][w] = Math.max(0, input[c][h][w]);
                }
            }
        }
    }
    
    /**
     * Convolution for a single filter using scalar operations
     */
    public static void convolveFilter(float[][][] padded, float[][][] kernels,
                                       float[][] output, int inputChannels,
                                       int kernelSize, int stride, float bias) {
        int outH = output.length;
        int outW = output[0].length;
        
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float sum = bias;
                
                for (int c = 0; c < inputChannels; c++) {
                    for (int kh = 0; kh < kernelSize; kh++) {
                        for (int kw = 0; kw < kernelSize; kw++) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            sum += padded[c][ih][iw] * kernels[c][kh][kw];
                        }
                    }
                }
                
                output[oh][ow] = sum;
            }
        }
    }
    
    /**
     * Max pooling for a single channel using scalar operations
     */
    public static void maxPool2D(float[][] input, float[][] output,
                                  int[][] maxIndicesH, int[][] maxIndicesW,
                                  int poolSize, int stride) {
        int outH = output.length;
        int outW = output[0].length;
        
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxH = 0, maxW = 0;
                
                for (int ph = 0; ph < poolSize; ph++) {
                    for (int pw = 0; pw < poolSize; pw++) {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        
                        if (input[ih][iw] > maxVal) {
                            maxVal = input[ih][iw];
                            maxH = ih;
                            maxW = iw;
                        }
                    }
                }
                
                output[oh][ow] = maxVal;
                maxIndicesH[oh][ow] = maxH;
                maxIndicesW[oh][ow] = maxW;
            }
        }
    }
}
