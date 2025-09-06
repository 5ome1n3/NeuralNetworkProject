#pragma once
#include <vector>
#include <array>
#include <exception>
#include <fstream>
#include <string>

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
	std::vector<std::vector<int>> ProcessingData;

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
		
		char character;
		std::string number;
		std::vector<int> row;
		row.reserve(785);
		ProcessingData.reserve(60000);
		ProcessingData.resize(60000);
		for (std::vector<int>& index : ProcessingData) {
			index.reserve(785);
		}
		int index = 0;
		while (TrainingData.get(character)) {
			if (character == ',') {
				row.push_back(std::stoi(number));
				
				number.clear();
			}
			else if (character == '\n') {
				ProcessingData[index] = row;
				row.clear();
				index++;
			}
			else if (character >= 48 && character <= 57) {
				number += character;
			}
			
		}
		TrainingData.close();
		std::cout << ProcessingData.size() << std::endl;
		std::cout << ProcessingData[34534].size() << std::endl;
	}
	void Train() {

	}
};