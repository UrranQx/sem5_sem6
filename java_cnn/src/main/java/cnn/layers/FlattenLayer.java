package cnn.layers;

import cnn.utils.Tensor3D;

/**
 * Flatten Layer
 * Converts 3D tensor to 1D array for fully connected layers
 */
public class FlattenLayer implements Layer {
    private int channels;
    private int height;
    private int width;
    
    @Override
    public Object forward(Object input) {
        Tensor3D inputTensor = (Tensor3D) input;
        channels = inputTensor.channels;
        height = inputTensor.height;
        width = inputTensor.width;
        
        return inputTensor.flatten();
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        float[] gradFlat = (float[]) gradient;
        return Tensor3D.fromFlattened(gradFlat, channels, height, width);
    }
    
    @Override
    public String getOutputShape() {
        return "Flatten";
    }
}
