package cnn.layers;

import cnn.utils.Tensor3D;

/**
 * ReLU Activation Layer
 */
public class ReLULayer implements Layer {
    private Tensor3D lastInput;
    private float[] lastInputFlat;
    private boolean is3D;
    
    @Override
    public Object forward(Object input) {
        if (input instanceof Tensor3D) {
            is3D = true;
            Tensor3D inputTensor = (Tensor3D) input;
            lastInput = inputTensor;
            return inputTensor.relu();
        } else {
            is3D = false;
            float[] inputArray = (float[]) input;
            lastInputFlat = inputArray;
            float[] output = new float[inputArray.length];
            for (int i = 0; i < inputArray.length; i++) {
                output[i] = Math.max(0, inputArray[i]);
            }
            return output;
        }
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        if (is3D) {
            Tensor3D gradOutput = (Tensor3D) gradient;
            Tensor3D gradInput = new Tensor3D(lastInput.channels, lastInput.height, lastInput.width);
            
            for (int c = 0; c < lastInput.channels; c++) {
                for (int h = 0; h < lastInput.height; h++) {
                    for (int w = 0; w < lastInput.width; w++) {
                        gradInput.data[c][h][w] = lastInput.data[c][h][w] > 0 ? 
                            gradOutput.data[c][h][w] : 0;
                    }
                }
            }
            return gradInput;
        } else {
            float[] gradOutput = (float[]) gradient;
            float[] gradInput = new float[lastInputFlat.length];
            for (int i = 0; i < lastInputFlat.length; i++) {
                gradInput[i] = lastInputFlat[i] > 0 ? gradOutput[i] : 0;
            }
            return gradInput;
        }
    }
    
    @Override
    public String getOutputShape() {
        return "ReLU";
    }
}
