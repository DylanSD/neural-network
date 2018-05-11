package com.neuralnetwork;

import com.pso.Gene;

/**
 * Created by dscottdawkins on 6/19/17.
 */

public class Network {

    private final int inputSize;
    private double[] hiddens;
    private double[] outputs;

    public Network(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        hiddens = new double[hiddenSize];
        outputs = new double[outputSize];
    }

    public int getWeightLength() {
        return inputSize * hiddens.length + hiddens.length * outputs.length;
    }

    public double[] runNetwork(double[] inputs, Gene gene) {
        long st = System.currentTimeMillis();
        int indx = 0;
        for (int i = 0 ; i < hiddens.length; i++) {
            double hid = 0;
            for (int j = 0 ; j < inputs.length; j++) {
                hid += inputs[j] * gene.getValue(indx++);
            }
            hiddens[i] = sigmoid(hid);
        }
        for (int i = 0 ; i < outputs.length; i++) {
            double hid = 0;
            for (int j = 0 ; j < hiddens.length; j++) {
                hid += hiddens[j] * gene.getValue(indx++);
            }
            outputs[i] = sigmoid(hid);
        }
        long ed = System.currentTimeMillis();
        //System.out.println("Time taken: " + (ed - st));
        return outputs;
    }

    private double sigmoid(double x) {
        return (1 / ( 1 + Math.pow(Math.E,(-1 * x))));
    }

}
