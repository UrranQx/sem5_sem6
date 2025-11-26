package cnn.utils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized tensor operations using Java Vector API
 * Provides SIMD-accelerated operations for CNN computations
 */
public class VectorOps {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    /**
     * Get the vector species being used
     */
    public static VectorSpecies<Float> getSpecies() {
        return SPECIES;
    }
    
    /**
     * Get the vector length (number of floats processed in parallel)
     */
    public static int getVectorLength() {
        return SPECIES.length();
    }
    
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
     * Uses Vector API for vectorized comparison
     */
    public static float[] reluDerivative(float[] a) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        FloatVector zero = FloatVector.zero(SPECIES);
        FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            // Create mask where elements > 0
            var mask = va.compare(jdk.incubator.vector.VectorOperators.GT, zero);
            // Select 1 where mask is true, 0 otherwise
            zero.blend(one, mask).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] > 0 ? 1.0f : 0.0f;
        }
        return result;
    }
    
    /**
     * Softmax activation
     * Note: Math.exp is not vectorizable in Vector API, so we use scalar operations
     * but vectorize the final normalization step
     */
    public static float[] softmax(float[] a) {
        float[] result = new float[a.length];
        float maxVal = max(a);
        
        // Compute exp(a[i] - max) - scalar since Math.exp is not vectorizable
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            result[i] = (float) Math.exp(a[i] - maxVal);
            sum += result[i];
        }
        
        // Vectorize the division
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        FloatVector divisor = FloatVector.broadcast(SPECIES, sum);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, result, i);
            va.div(divisor).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
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
    
    /**
     * Add value to array in-place using Vector API
     */
    public static void addInPlace(float[] a, float[] b) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(a, i);
        }
        
        for (; i < a.length; i++) {
            a[i] += b[i];
        }
    }
    
    /**
     * Scale array in-place using Vector API
     */
    public static void scaleInPlace(float[] a, float scalar) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            va.mul(scalar).intoArray(a, i);
        }
        
        for (; i < a.length; i++) {
            a[i] *= scalar;
        }
    }
    
    /**
     * Subtract and scale array in-place (a -= scalar * b) using Vector API
     * Used for gradient descent weight updates
     */
    public static void subtractScaledInPlace(float[] a, float[] b, float scalar) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.sub(vb.mul(scalar)).intoArray(a, i);
        }
        
        for (; i < a.length; i++) {
            a[i] -= scalar * b[i];
        }
    }
    
    /**
     * ReLU activation in-place using Vector API
     */
    public static void reluInPlace(float[] a) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            va.max(zero).intoArray(a, i);
        }
        
        for (; i < a.length; i++) {
            a[i] = Math.max(0, a[i]);
        }
    }
    
    /**
     * Compute convolution sum for a single output position using Vector API
     * This vectorizes the innermost kernel loops (kh, kw) when kernel size is large enough
     * 
     * @param padded Padded input data [channels][height][width]
     * @param kernels Kernel weights [inputChannels][kernelH][kernelW]
     * @param inputChannels Number of input channels
     * @param kernelSize Size of kernel (assumes square kernel)
     * @param startH Starting height position in padded input
     * @param startW Starting width position in padded input
     * @param bias Bias value to add
     * @return Convolution sum for this output position
     */
    public static float convolve(float[][][] padded, float[][][] kernels, 
                                  int inputChannels, int kernelSize,
                                  int startH, int startW, float bias) {
        int kernelArea = kernelSize * kernelSize;
        float sum = bias;
        
        // Process each input channel
        for (int c = 0; c < inputChannels; c++) {
            // Flatten the kernel region for vectorized dot product
            int i = 0;
            int upperBound = SPECIES.loopBound(kernelArea);
            FloatVector sumVector = FloatVector.zero(SPECIES);
            
            // Vectorized processing of kernel elements
            for (; i < upperBound; i += SPECIES.length()) {
                // Load kernel values
                float[] kernelChunk = new float[SPECIES.length()];
                float[] inputChunk = new float[SPECIES.length()];
                
                for (int j = 0; j < SPECIES.length(); j++) {
                    int kIdx = i + j;
                    int kh = kIdx / kernelSize;
                    int kw = kIdx % kernelSize;
                    kernelChunk[j] = kernels[c][kh][kw];
                    inputChunk[j] = padded[c][startH + kh][startW + kw];
                }
                
                FloatVector vk = FloatVector.fromArray(SPECIES, kernelChunk, 0);
                FloatVector vi = FloatVector.fromArray(SPECIES, inputChunk, 0);
                sumVector = sumVector.add(vk.mul(vi));
            }
            
            sum += sumVector.reduceLanes(VectorOperators.ADD);
            
            // Handle remaining elements
            for (; i < kernelArea; i++) {
                int kh = i / kernelSize;
                int kw = i % kernelSize;
                sum += padded[c][startH + kh][startW + kw] * kernels[c][kh][kw];
            }
        }
        
        return sum;
    }
    
    /**
     * Vectorized convolution forward pass for a single filter
     * Processes multiple output positions simultaneously when possible
     * Optimized for stride=1 case which is most common
     * 
     * @param padded Padded input [channels][height][width]
     * @param kernels Kernel weights for one filter [inputChannels][kernelH][kernelW]
     * @param output Output array for this filter [outH][outW]
     * @param inputChannels Number of input channels
     * @param kernelSize Kernel size
     * @param stride Convolution stride
     * @param bias Bias for this filter
     */
    public static void convolveFilter(float[][][] padded, float[][][] kernels,
                                       float[][] output, int inputChannels,
                                       int kernelSize, int stride, float bias) {
        int outH = output.length;
        int outW = output[0].length;
        int vectorLen = SPECIES.length();
        
        // Process output positions
        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * stride;
            
            // Vectorize across output width where possible
            int ow = 0;
            int upperBound = SPECIES.loopBound(outW);
            
            // Optimized for stride=1: can use direct array loading
            if (stride == 1) {
                for (; ow < upperBound; ow += vectorLen) {
                    FloatVector result = FloatVector.broadcast(SPECIES, bias);
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            int inputH = ih + kh;
                            
                            for (int kw = 0; kw < kernelSize; kw++) {
                                float kernelVal = kernels[c][kh][kw];
                                int inputW = ow + kw;
                                
                                // Direct load - efficient for stride=1
                                FloatVector inputVec = FloatVector.fromArray(SPECIES, padded[c][inputH], inputW);
                                result = result.add(inputVec.mul(kernelVal));
                            }
                        }
                    }
                    
                    result.intoArray(output[oh], ow);
                }
            } else {
                // Generic case for stride > 1
                for (; ow < upperBound; ow += vectorLen) {
                    float[] results = new float[vectorLen];
                    
                    for (int v = 0; v < vectorLen; v++) {
                        results[v] = bias;
                    }
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                float kernelVal = kernels[c][kh][kw];
                                
                                float[] inputVals = new float[vectorLen];
                                for (int v = 0; v < vectorLen; v++) {
                                    int iw = (ow + v) * stride + kw;
                                    inputVals[v] = padded[c][ih + kh][iw];
                                }
                                
                                FloatVector vi = FloatVector.fromArray(SPECIES, inputVals, 0);
                                FloatVector vr = FloatVector.fromArray(SPECIES, results, 0);
                                vr.add(vi.mul(kernelVal)).intoArray(results, 0);
                            }
                        }
                    }
                    
                    System.arraycopy(results, 0, output[oh], ow, vectorLen);
                }
            }
            
            // Handle remaining positions
            for (; ow < outW; ow++) {
                int iw = ow * stride;
                float sum = bias;
                
                for (int c = 0; c < inputChannels; c++) {
                    for (int kh = 0; kh < kernelSize; kh++) {
                        for (int kw = 0; kw < kernelSize; kw++) {
                            sum += padded[c][ih + kh][iw + kw] * kernels[c][kh][kw];
                        }
                    }
                }
                
                output[oh][ow] = sum;
            }
        }
    }
    
    /**
     * Vectorized max pooling for a single channel
     * 
     * @param input Input channel data [height][width]
     * @param output Output channel data [outH][outW]
     * @param maxIndicesH Output array for max indices (height) [outH][outW]
     * @param maxIndicesW Output array for max indices (width) [outH][outW]
     * @param poolSize Pool window size
     * @param stride Pool stride
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
                
                int poolArea = poolSize * poolSize;
                int startH = oh * stride;
                int startW = ow * stride;
                
                // For small pool sizes, use scalar code
                if (poolArea < SPECIES.length()) {
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = startH + ph;
                            int iw = startW + pw;
                            if (input[ih][iw] > maxVal) {
                                maxVal = input[ih][iw];
                                maxH = ih;
                                maxW = iw;
                            }
                        }
                    }
                } else {
                    // Vectorized max for larger pool sizes
                    float[] poolVals = new float[poolArea];
                    int idx = 0;
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            poolVals[idx++] = input[startH + ph][startW + pw];
                        }
                    }
                    
                    // Find max using vectors
                    int i = 0;
                    int upperBound = SPECIES.loopBound(poolArea);
                    
                    if (poolArea >= SPECIES.length()) {
                        FloatVector maxVector = FloatVector.fromArray(SPECIES, poolVals, 0);
                        for (i = SPECIES.length(); i < upperBound; i += SPECIES.length()) {
                            FloatVector v = FloatVector.fromArray(SPECIES, poolVals, i);
                            maxVector = maxVector.max(v);
                        }
                        maxVal = maxVector.reduceLanes(VectorOperators.MAX);
                    }
                    
                    // Handle tail
                    for (; i < poolArea; i++) {
                        if (poolVals[i] > maxVal) {
                            maxVal = poolVals[i];
                        }
                    }
                    
                    // Find the index of max value (needed for backprop)
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = startH + ph;
                            int iw = startW + pw;
                            if (input[ih][iw] == maxVal) {
                                maxH = ih;
                                maxW = iw;
                                break;
                            }
                        }
                    }
                }
                
                output[oh][ow] = maxVal;
                maxIndicesH[oh][ow] = maxH;
                maxIndicesW[oh][ow] = maxW;
            }
        }
    }
    
    /**
     * Vectorized ReLU for 3D tensor
     * 
     * @param input Input tensor data [channels][height][width]
     * @param output Output tensor data [channels][height][width]
     */
    public static void relu3D(float[][][] input, float[][][] output) {
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, input[c][h], w);
                    v.max(zero).intoArray(output[c][h], w);
                }
                
                for (; w < width; w++) {
                    output[c][h][w] = Math.max(0, input[c][h][w]);
                }
            }
        }
    }
    
    /**
     * Vectorized ReLU derivative for 3D tensor backpropagation
     * Computes: gradOutput * (input > 0 ? 1 : 0)
     * 
     * @param input Original input tensor [channels][height][width]
     * @param gradOutput Gradient from next layer [channels][height][width]
     * @param gradInput Output gradient for previous layer [channels][height][width]
     */
    public static void reluBackward3D(float[][][] input, float[][][] gradOutput, float[][][] gradInput) {
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(width);
                
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector vi = FloatVector.fromArray(SPECIES, input[c][h], w);
                    FloatVector vg = FloatVector.fromArray(SPECIES, gradOutput[c][h], w);
                    
                    // Create mask where input > 0
                    VectorMask<Float> mask = vi.compare(VectorOperators.GT, zero);
                    // Select gradient where mask is true, 0 otherwise
                    zero.blend(vg, mask).intoArray(gradInput[c][h], w);
                }
                
                for (; w < width; w++) {
                    gradInput[c][h][w] = input[c][h][w] > 0 ? gradOutput[c][h][w] : 0;
                }
            }
        }
    }
    
    /**
     * Vectorized ReLU derivative for 1D array backpropagation
     * 
     * @param input Original input array
     * @param gradOutput Gradient from next layer
     * @return Gradient for previous layer
     */
    public static float[] reluBackward(float[] input, float[] gradOutput) {
        float[] gradInput = new float[input.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(input.length);
        
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector vi = FloatVector.fromArray(SPECIES, input, i);
            FloatVector vg = FloatVector.fromArray(SPECIES, gradOutput, i);
            
            VectorMask<Float> mask = vi.compare(VectorOperators.GT, zero);
            zero.blend(vg, mask).intoArray(gradInput, i);
        }
        
        for (; i < input.length; i++) {
            gradInput[i] = input[i] > 0 ? gradOutput[i] : 0;
        }
        
        return gradInput;
    }
    
    /**
     * Vectorized array fill with value
     */
    public static void fill(float[] array, float value) {
        int i = 0;
        int upperBound = SPECIES.loopBound(array.length);
        FloatVector fillVector = FloatVector.broadcast(SPECIES, value);
        
        for (; i < upperBound; i += SPECIES.length()) {
            fillVector.intoArray(array, i);
        }
        
        for (; i < array.length; i++) {
            array[i] = value;
        }
    }
    
    /**
     * Vectorized array copy
     */
    public static void copy(float[] src, float[] dest) {
        int i = 0;
        int upperBound = SPECIES.loopBound(src.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, src, i);
            v.intoArray(dest, i);
        }
        
        for (; i < src.length; i++) {
            dest[i] = src[i];
        }
    }
    
    /**
     * Fused multiply-add: result[i] = a[i] + b[i] * c
     */
    public static float[] fma(float[] a, float[] b, float c) {
        float[] result = new float[a.length];
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb.mul(c)).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] + b[i] * c;
        }
        
        return result;
    }
    
    /**
     * Fused multiply-add in-place: a[i] += b[i] * c
     */
    public static void fmaInPlace(float[] a, float[] b, float c) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb.mul(c)).intoArray(a, i);
        }
        
        for (; i < a.length; i++) {
            a[i] += b[i] * c;
        }
    }
    
    /**
     * Outer product: result[i][j] = a[i] * b[j]
     * Used for computing weight gradients in dense layers
     */
    public static void outerProductAddScaled(float[][] result, float[] a, float[] b, float scale) {
        for (int i = 0; i < a.length; i++) {
            float ai = a[i] * scale;
            
            int j = 0;
            int upperBound = SPECIES.loopBound(b.length);
            FloatVector scaledA = FloatVector.broadcast(SPECIES, ai);
            
            for (; j < upperBound; j += SPECIES.length()) {
                FloatVector vr = FloatVector.fromArray(SPECIES, result[i], j);
                FloatVector vb = FloatVector.fromArray(SPECIES, b, j);
                vr.add(scaledA.mul(vb)).intoArray(result[i], j);
            }
            
            for (; j < b.length; j++) {
                result[i][j] += ai * b[j];
            }
        }
    }
}
