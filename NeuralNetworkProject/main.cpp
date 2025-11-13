#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"


int main(int argc, char* argv[]) {
    srand(time(0));
    const int InputSize = 784;
    const int OutputSize = 10;
    const int LayerCount = 2;
    const int LayerSize = 16;
    const int DataRows = 10000;
    double LearningRate = 0.0005;
    NeuralNetwork<InputSize, OutputSize, LayerSize, LayerCount> nn(LearningRate, RELU);
    if (!nn.LoadData("mnist_train.csv", DataRows, true)) {
        return -1;
    }
    nn.Train();
    
    if (!nn.LoadData("mnist_test.csv", 10000, true)) {
        return -1;
    }
    nn.Run();
    nn.PrintOutput();
    return 0;

}