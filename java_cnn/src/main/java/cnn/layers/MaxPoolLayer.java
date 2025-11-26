package cnn.layers;

import cnn.utils.Tensor3D;
import cnn.utils.VectorOps;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Max Pooling Layer with Java Vector API optimization
 * Reduces spatial dimensions by taking max value in each pooling window
 * Uses SIMD vectorization for accelerated computation
 */
public class MaxPoolLayer implements Layer {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private int poolSize;
    private int stride;
    
    private Tensor3D lastInput;
    private int[][][] maxIndicesH;  // Store indices for backprop
    private int[][][] maxIndicesW;
    
    public MaxPoolLayer(int poolSize) {
        this(poolSize, poolSize);
    }
    
    public MaxPoolLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }
    
    @Override
    public Object forward(Object input) {
        Tensor3D inputTensor = (Tensor3D) input;
        lastInput = inputTensor;
        
        int outH = (inputTensor.height - poolSize) / stride + 1;
        int outW = (inputTensor.width - poolSize) / stride + 1;
        
        Tensor3D output = new Tensor3D(inputTensor.channels, outH, outW);
        maxIndicesH = new int[inputTensor.channels][outH][outW];
        maxIndicesW = new int[inputTensor.channels][outH][outW];
        
        // Vectorized max pooling for each channel
        for (int c = 0; c < inputTensor.channels; c++) {
            vectorizedMaxPool(inputTensor.data[c], output.data[c], 
                            maxIndicesH[c], maxIndicesW[c], outH, outW);
        }
        
        return output;
    }
    
    /**
     * Vectorized max pooling for a single channel
     * Uses SIMD operations to find maximum values
     */
    private void vectorizedMaxPool(float[][] input, float[][] output,
                                   int[][] indicesH, int[][] indicesW,
                                   int outH, int outW) {
        int poolArea = poolSize * poolSize;
        
        for (int oh = 0; oh < outH; oh++) {
            int startH = oh * stride;
            
            for (int ow = 0; ow < outW; ow++) {
                int startW = ow * stride;
                
                // For typical small pool sizes (2x2, 3x3), vectorization of the pool window
                // doesn't help much, but we can still use vectorized max finding
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxH = startH;
                int maxW = startW;
                
                if (poolArea >= SPECIES.length()) {
                    // Collect all values in the pool window
                    float[] poolVals = new float[poolArea];
                    int idx = 0;
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            poolVals[idx++] = input[startH + ph][startW + pw];
                        }
                    }
                    
                    // Vectorized max finding
                    int i = 0;
                    int upperBound = SPECIES.loopBound(poolArea);
                    
                    if (poolArea >= SPECIES.length()) {
                        FloatVector maxVec = FloatVector.fromArray(SPECIES, poolVals, 0);
                        for (i = SPECIES.length(); i < upperBound; i += SPECIES.length()) {
                            FloatVector v = FloatVector.fromArray(SPECIES, poolVals, i);
                            maxVec = maxVec.max(v);
                        }
                        maxVal = maxVec.reduceLanes(VectorOperators.MAX);
                    }
                    
                    // Handle remaining elements
                    for (; i < poolArea; i++) {
                        if (poolVals[i] > maxVal) {
                            maxVal = poolVals[i];
                        }
                    }
                    
                    // Find the position of the maximum value (needed for backprop)
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            if (input[startH + ph][startW + pw] == maxVal) {
                                maxH = startH + ph;
                                maxW = startW + pw;
                                ph = poolSize; // break outer loop
                                break;
                            }
                        }
                    }
                } else {
                    // Scalar max for small pool sizes (typical case: 2x2)
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
                }
                
                output[oh][ow] = maxVal;
                indicesH[oh][ow] = maxH;
                indicesW[oh][ow] = maxW;
            }
        }
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        Tensor3D gradOutput = (Tensor3D) gradient;
        Tensor3D gradInput = new Tensor3D(lastInput.channels, lastInput.height, lastInput.width);
        
        // Backward pass - route gradients to max positions
        // This is inherently sparse and doesn't benefit much from vectorization
        for (int c = 0; c < gradOutput.channels; c++) {
            for (int oh = 0; oh < gradOutput.height; oh++) {
                for (int ow = 0; ow < gradOutput.width; ow++) {
                    int ih = maxIndicesH[c][oh][ow];
                    int iw = maxIndicesW[c][oh][ow];
                    gradInput.data[c][ih][iw] += gradOutput.data[c][oh][ow];
                }
            }
        }
        
        return gradInput;
    }
    
    @Override
    public String getOutputShape() {
        return "MaxPool2D(" + poolSize + "x" + poolSize + ")";
    }
}
