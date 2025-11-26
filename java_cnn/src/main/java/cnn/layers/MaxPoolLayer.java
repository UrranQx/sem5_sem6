package cnn.layers;

import cnn.utils.Tensor3D;

/**
 * Max Pooling Layer
 * Reduces spatial dimensions by taking max value in each pooling window
 */
public class MaxPoolLayer implements Layer {
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
        
        for (int c = 0; c < inputTensor.channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float maxVal = Float.NEGATIVE_INFINITY;
                    int maxH = 0, maxW = 0;
                    
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            
                            if (inputTensor.data[c][ih][iw] > maxVal) {
                                maxVal = inputTensor.data[c][ih][iw];
                                maxH = ih;
                                maxW = iw;
                            }
                        }
                    }
                    
                    output.data[c][oh][ow] = maxVal;
                    maxIndicesH[c][oh][ow] = maxH;
                    maxIndicesW[c][oh][ow] = maxW;
                }
            }
        }
        
        return output;
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        Tensor3D gradOutput = (Tensor3D) gradient;
        Tensor3D gradInput = new Tensor3D(lastInput.channels, lastInput.height, lastInput.width);
        
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
