#pragma once
#include <vector>
#include <array>
#include <exception>
#include <fstream>

#include "Matrix.hpp"
#include "Vector.hpp"
#include "NetworkLayer.hpp"


class BaseException : public std::logic_error {
public:
	BaseException(const char* msg)
		:std::logic_error(msg)
		{}
};

class HiddenLayerCountException : public BaseException {
public:
	HiddenLayerCountException(const char* msg)
		:BaseException(msg)
		{}
	
};

using count_t = unsigned long long;

template<size_t InputSize, size_t OutputSize, size_t HiddenLayerSize, count_t HiddenLayerCount>
class NeuralNetwork {
private:
	double InitialLearningRate;
	Layer<InputSize, 0> InputLayer;
	Layer<OutputSize, HiddenLayerSize> OutputLayer;
	Layer<HiddenLayerSize, InputSize> FirstLayer;
	std::vector<Layer<HiddenLayerSize, HiddenLayerSize>> HiddenLayers;
	std::vector<double> ProcessingData;

	void BackPropActivation() {

	}
	
public:
	NeuralNetwork(double InitialLearningRate, ActivationType type)
		:InitialLearningRate(InitialLearningRate)
	{
		if (HiddenLayerCount < 1) {
			throw HiddenLayerCountException("hidden layer count must be greater than 1");
		}
		HiddenLayers.reserve(HiddenLayerCount - 1);
		HiddenLayers.resize(HiddenLayerCount - 1);


	}
	NeuralNetwork(double InitialLearningRate, ActivationType type, const char* )
		:InitialLearningRate(InitialLearningRate)
	{
		if (HiddenLayerCount < 1) {
			throw HiddenLayerCountException("hidden layer count must be greater than 1");
		}
		HiddenLayers.reserve(HiddenLayerCount - 1);
		HiddenLayers.resize(HiddenLayerCount - 1);


	}
	~NeuralNetwork() {

	}

	void PropogateForward() {
		FirstLayer.CalcForwardProp(InputLayer);


		for (int index = 1; index < HiddenLayers.size();index++) {
			HiddenLayers[index].CalcForwardProp(HiddenLayers[index - 1]);
		}

		if (HiddenLayerCount == 1) {
			OutputLayer.CalcForwardProp(FirstLayer);
			return;
		}

		if (HiddenLayerCount == 2) {
			HiddenLayers[0].CalcForwardProp(FirstLayer);
		}
		OutputLayer.CalcForwardProp(HiddenLayers.back());
	}
	void PropogateBackward() {

	}
	void LoadTrainingData(const char* FileName) {
		std::ifstream TrainingData(FileName);
		std::vector<std::vector<double>> whole;
		std::string TrainingInput;
		std::string value;
		while (getline(TrainingData, TrainingInput, '\n')) {
			std::vector<double> line;
			for (const char& digit : TrainingInput) {
				if (digit == ',') {
					line.push_back(std::stoi(value));
					value = "";
				}
				else {
					value += digit;
				}

			}
			
			line.push_back(std::stoi(value));
			whole.push_back(line);
		}
		std::cout << whole.size() << std::endl;
		//std::cout << whole[0].size() << std::endl;
		TrainingData.close();
	}
	void Train() {

	}
};