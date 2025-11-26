package cnn.layers;

import cnn.utils.Tensor3D;
import cnn.utils.VectorOps;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * 2D Convolution Layer with Java Vector API optimization
 * Implements forward and backward pass for convolution operation
 * Uses SIMD vectorization for accelerated computation
 */
public class ConvLayer implements Layer {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private float[][][][] kernels;  // [numFilters][inputChannels][kernelH][kernelW]
    private float[] biases;
    private int numFilters;
    private int kernelSize;
    private int stride;
    private int padding;
    private int inputChannels;
    
    private Tensor3D lastInput;
    private Tensor3D lastOutput;
    
    public ConvLayer(int inputChannels, int numFilters, int kernelSize, int stride, int padding) {
        this.inputChannels = inputChannels;
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        
        initializeWeights();
    }
    
    private void initializeWeights() {
        Random rand = new Random(42);
        // He initialization
        float scale = (float) Math.sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        
        kernels = new float[numFilters][inputChannels][kernelSize][kernelSize];
        biases = new float[numFilters];
        
        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernels[f][c][i][j] = (float) (rand.nextGaussian() * scale);
                    }
                }
            }
            biases[f] = 0.01f;
        }
    }
    
    @Override
    public Object forward(Object input) {
        Tensor3D inputTensor = (Tensor3D) input;
        lastInput = inputTensor;
        
        // Apply padding
        Tensor3D padded = inputTensor.pad(padding);
        
        int outH = (padded.height - kernelSize) / stride + 1;
        int outW = (padded.width - kernelSize) / stride + 1;
        
        Tensor3D output = new Tensor3D(numFilters, outH, outW);
        
        // Vectorized convolution for each filter
        for (int f = 0; f < numFilters; f++) {
            vectorizedConvolveFilter(padded.data, kernels[f], output.data[f], biases[f], outH, outW);
        }
        
        lastOutput = output;
        return output;
    }
    
    /**
     * Vectorized convolution for a single filter
     * Uses SIMD to process multiple output positions in parallel
     * Optimized for stride=1 case which is most common
     */
    private void vectorizedConvolveFilter(float[][][] padded, float[][][] kernel, 
                                           float[][] output, float bias, int outH, int outW) {
        int vectorLen = SPECIES.length();
        int paddedW = padded[0][0].length;
        
        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * stride;
            
            // Process output width with vectorization
            int ow = 0;
            
            // Vectorized processing - optimized for stride=1
            if (stride == 1) {
                // Calculate safe upper bound - ensure we don't read past padded array
                // For stride=1: we read from position (ow + kw) to (ow + kw + vectorLen - 1)
                int safeUpperBound = Math.min(SPECIES.loopBound(outW), 
                                              paddedW - kernelSize - vectorLen + 2);
                safeUpperBound = Math.max(0, safeUpperBound);
                
                for (; ow < safeUpperBound; ow += vectorLen) {
                    // Initialize result vectors with bias
                    FloatVector result = FloatVector.broadcast(SPECIES, bias);
                    
                    // Accumulate convolution for all input channels and kernel positions
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            int inputH = ih + kh;
                            
                            // For stride=1, consecutive output positions use consecutive input positions
                            // This allows direct vectorized loading
                            for (int kw = 0; kw < kernelSize; kw++) {
                                float kernelVal = kernel[c][kh][kw];
                                int inputW = ow + kw;
                                
                                // Direct load - efficient for stride=1
                                FloatVector inputVec = FloatVector.fromArray(SPECIES, padded[c][inputH], inputW);
                                result = result.add(inputVec.mul(kernelVal));
                            }
                        }
                    }
                    
                    // Store results
                    result.intoArray(output[oh], ow);
                }
            } else {
                // Generic case for stride > 1 (with gather)
                int upperBound = SPECIES.loopBound(outW);
                for (; ow < upperBound; ow += vectorLen) {
                    // Initialize result vectors with bias
                    FloatVector result = FloatVector.broadcast(SPECIES, bias);
                    
                    // Accumulate convolution for all input channels and kernel positions
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            int inputH = ih + kh;
                            
                            for (int kw = 0; kw < kernelSize; kw++) {
                                float kernelVal = kernel[c][kh][kw];
                                
                                // Gather input values for vector positions (slower for stride > 1)
                                float[] inputVals = new float[vectorLen];
                                for (int v = 0; v < vectorLen; v++) {
                                    int inputW = (ow + v) * stride + kw;
                                    inputVals[v] = padded[c][inputH][inputW];
                                }
                                
                                FloatVector inputVec = FloatVector.fromArray(SPECIES, inputVals, 0);
                                result = result.add(inputVec.mul(kernelVal));
                            }
                        }
                    }
                    
                    // Store results
                    result.intoArray(output[oh], ow);
                }
            }
            
            // Handle remaining positions with scalar code
            for (; ow < outW; ow++) {
                float sum = bias;
                int iw = ow * stride;
                
                for (int c = 0; c < inputChannels; c++) {
                    for (int kh = 0; kh < kernelSize; kh++) {
                        for (int kw = 0; kw < kernelSize; kw++) {
                            sum += padded[c][ih + kh][iw + kw] * kernel[c][kh][kw];
                        }
                    }
                }
                
                output[oh][ow] = sum;
            }
        }
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        Tensor3D gradOutput = (Tensor3D) gradient;
        
        // Compute gradient for previous layer
        Tensor3D padded = lastInput.pad(padding);
        Tensor3D gradInput = new Tensor3D(inputChannels, lastInput.height, lastInput.width);
        
        float[][][][] gradKernels = new float[numFilters][inputChannels][kernelSize][kernelSize];
        float[] gradBiases = new float[numFilters];
        
        // Calculate gradients with vectorization where possible
        for (int f = 0; f < numFilters; f++) {
            vectorizedBackward(f, padded.data, gradOutput.data[f], gradInput.data, 
                              gradKernels[f], gradBiases, f);
        }
        
        // Update weights using vectorized operations
        for (int f = 0; f < numFilters; f++) {
            biases[f] -= learningRate * gradBiases[f];
            for (int c = 0; c < inputChannels; c++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    // Vectorize kernel weight updates where possible
                    int kw = 0;
                    int upperBound = SPECIES.loopBound(kernelSize);
                    
                    for (; kw < upperBound; kw += SPECIES.length()) {
                        FloatVector vk = FloatVector.fromArray(SPECIES, kernels[f][c][kh], kw);
                        FloatVector vg = FloatVector.fromArray(SPECIES, gradKernels[f][c][kh], kw);
                        vk.sub(vg.mul(learningRate)).intoArray(kernels[f][c][kh], kw);
                    }
                    
                    for (; kw < kernelSize; kw++) {
                        kernels[f][c][kh][kw] -= learningRate * gradKernels[f][c][kh][kw];
                    }
                }
            }
        }
        
        return gradInput;
    }
    
    /**
     * Vectorized backward pass for a single filter
     */
    private void vectorizedBackward(int filterIdx, float[][][] padded, float[][] gradOut,
                                    float[][][] gradInput, float[][][] gradKernels,
                                    float[] gradBiases, int biasIdx) {
        int outH = gradOut.length;
        int outW = gradOut[0].length;
        
        // Vectorized bias gradient accumulation
        for (int oh = 0; oh < outH; oh++) {
            int ow = 0;
            int upperBound = SPECIES.loopBound(outW);
            FloatVector sumVec = FloatVector.zero(SPECIES);
            
            for (; ow < upperBound; ow += SPECIES.length()) {
                FloatVector vg = FloatVector.fromArray(SPECIES, gradOut[oh], ow);
                sumVec = sumVec.add(vg);
            }
            
            gradBiases[biasIdx] += sumVec.reduceLanes(VectorOperators.ADD);
            
            for (; ow < outW; ow++) {
                gradBiases[biasIdx] += gradOut[oh][ow];
            }
        }
        
        // Compute kernel gradients and input gradients
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float grad = gradOut[oh][ow];
                
                for (int c = 0; c < inputChannels; c++) {
                    for (int kh = 0; kh < kernelSize; kh++) {
                        int ih = oh * stride + kh;
                        
                        for (int kw = 0; kw < kernelSize; kw++) {
                            int iw = ow * stride + kw;
                            
                            gradKernels[c][kh][kw] += padded[c][ih][iw] * grad;
                            
                            // Gradient for input (need to handle padding)
                            int origH = ih - padding;
                            int origW = iw - padding;
                            if (origH >= 0 && origH < lastInput.height && 
                                origW >= 0 && origW < lastInput.width) {
                                gradInput[c][origH][origW] += kernels[filterIdx][c][kh][kw] * grad;
                            }
                        }
                    }
                }
            }
        }
    }
    
    @Override
    public String getOutputShape() {
        return "Conv2D(" + numFilters + " filters, " + kernelSize + "x" + kernelSize + ")";
    }
}
