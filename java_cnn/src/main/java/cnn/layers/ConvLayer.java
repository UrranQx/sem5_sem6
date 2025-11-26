package cnn.layers;

import cnn.utils.Tensor3D;
import cnn.utils.VectorOps;
import java.util.Random;

/**
 * 2D Convolution Layer
 * Implements forward and backward pass for convolution operation
 */
public class ConvLayer implements Layer {
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
        
        // Convolution operation
        for (int f = 0; f < numFilters; f++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = biases[f];
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                sum += padded.data[c][ih][iw] * kernels[f][c][kh][kw];
                            }
                        }
                    }
                    
                    output.data[f][oh][ow] = sum;
                }
            }
        }
        
        lastOutput = output;
        return output;
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        Tensor3D gradOutput = (Tensor3D) gradient;
        
        // Compute gradient for previous layer
        Tensor3D padded = lastInput.pad(padding);
        Tensor3D gradInput = new Tensor3D(inputChannels, lastInput.height, lastInput.width);
        
        float[][][][] gradKernels = new float[numFilters][inputChannels][kernelSize][kernelSize];
        float[] gradBiases = new float[numFilters];
        
        // Calculate gradients
        for (int f = 0; f < numFilters; f++) {
            for (int oh = 0; oh < gradOutput.height; oh++) {
                for (int ow = 0; ow < gradOutput.width; ow++) {
                    float grad = gradOutput.data[f][oh][ow];
                    gradBiases[f] += grad;
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                
                                gradKernels[f][c][kh][kw] += padded.data[c][ih][iw] * grad;
                                
                                // Gradient for input (need to handle padding)
                                int origH = ih - padding;
                                int origW = iw - padding;
                                if (origH >= 0 && origH < lastInput.height && 
                                    origW >= 0 && origW < lastInput.width) {
                                    gradInput.data[c][origH][origW] += kernels[f][c][kh][kw] * grad;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Update weights
        for (int f = 0; f < numFilters; f++) {
            biases[f] -= learningRate * gradBiases[f];
            for (int c = 0; c < inputChannels; c++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < kernelSize; kw++) {
                        kernels[f][c][kh][kw] -= learningRate * gradKernels[f][c][kh][kw];
                    }
                }
            }
        }
        
        return gradInput;
    }
    
    @Override
    public String getOutputShape() {
        return "Conv2D(" + numFilters + " filters, " + kernelSize + "x" + kernelSize + ")";
    }
}
