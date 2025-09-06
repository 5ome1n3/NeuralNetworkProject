#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"


int main(int argc, char* argv[]) {
    const int InputSize = 784;
    const int OutputSize = 10;
    const int LayerCount = 5;
    const int LayerSize = 16;
    int LearningRate = 0.5;
    NeuralNetwork<InputSize, OutputSize, LayerSize, LayerCount> nn(LearningRate, SIGMOID);
    
    nn.LoadTrainingData("mnist_train.csv");

    nn.PropogateForward();

}