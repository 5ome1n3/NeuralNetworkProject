#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"



int main(int argc, char* argv[]) {
    srand(time(0));
    std::cout << rand() << std::endl;
    const int InputSize = 784;
    const int OutputSize = 10;
    const int LayerCount = 2;
    const int LayerSize = 16;
    const int DataRows = 10000;
    double LearningRate = 0.0005;
    NeuralNetwork<InputSize, OutputSize, LayerSize, LayerCount> nn(LearningRate, RELU);
    
    nn.LoadData("mnist_train.csv", DataRows, false);
    nn.Train();
    nn.LoadData("mnist_test.csv", 10000, false);
    nn.Run();
    nn.PrintOutput();
    

}