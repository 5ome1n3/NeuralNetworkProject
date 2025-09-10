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

template<size_t size>
int GetMaxValue(Vector<size> vector) {
	double value = 0;
	for (int index = 0; index < vector.GetSize(); index++) {
		value = std::max(value, vector[index]);
	}
	return value;
}

using count_t = unsigned long long;

template<const size_t InputSize, const size_t OutputSize, const size_t HiddenLayerSize, const count_t HiddenLayerCount>
class NeuralNetwork {
private:
	double InitialLearningRate;
	Layer<InputSize, 0> InputLayer;
	Layer<OutputSize, HiddenLayerSize> OutputLayer;
	Layer<HiddenLayerSize, InputSize> FirstHiddenLayer;
	std::vector<Layer<HiddenLayerSize, HiddenLayerSize>> HiddenLayers;
	std::vector<std::vector<double>> ProcessingData;
	int CurrentTargetIndex;
	int count = 0;
	void BackPropActivations() {
		// calculates error delta for output layer
		Vector<OutputSize> OutputLayerDerivatives = SigmoidifiedDerivative(OutputLayer.GetLayerNeurons());
		Vector<OutputSize> ErrorDeltas;
		for (int index = 0; index < ErrorDeltas.GetSize(); index++) {
			if (index == CurrentTargetIndex) {
				ErrorDeltas[index] = 2 * (OutputLayer.GetLayerActivatedNeurons()[index] - 1);
			}
			else {
				ErrorDeltas[index] = 2 * (OutputLayer.GetLayerActivatedNeurons()[index]);
			}
		}
		OutputLayer.SetLayerErrorDeltas(ErrorDeltas * OutputLayerDerivatives);

		if (HiddenLayerCount > 1) {
			// calculates error deltas for rest of layers
			// next layer weights, cur layer error deltas, cur layer neurons, next layer error deltas
			for (int index = HiddenLayerCount - 2; index >= 0; index--) {
				Vector<HiddenLayerSize> CurrentLayerDerivatives = SigmoidifiedDerivative(HiddenLayers[index].GetLayerNeurons());
				if (index == HiddenLayerCount - 2) {
					HiddenLayers[index].SetLayerErrorDeltas((OutputLayer.GetLayerWeights().GetTransposed() * OutputLayer.GetLayerErrorDeltas()) * CurrentLayerDerivatives);
				}
				else {
					HiddenLayers[index].SetLayerErrorDeltas((HiddenLayers[index + 1].GetLayerWeights().GetTransposed() * HiddenLayers[index + 1].GetLayerErrorDeltas()) * CurrentLayerDerivatives);
				}
			}
			Vector<HiddenLayerSize> CurrentLayerDerivatives = SigmoidifiedDerivative(FirstHiddenLayer.GetLayerNeurons());
			FirstHiddenLayer.SetLayerErrorDeltas((HiddenLayers[0].GetLayerWeights().GetTransposed() * HiddenLayers[0].GetLayerErrorDeltas()) * CurrentLayerDerivatives);
		}
		else {
			Vector<HiddenLayerSize> CurrentLayerDerivatives = SigmoidifiedDerivative(FirstHiddenLayer.GetLayerNeurons());
			FirstHiddenLayer.SetLayerErrorDeltas((OutputLayer.GetLayerWeights().GetTransposed() * OutputLayer.GetLayerErrorDeltas()) * CurrentLayerDerivatives);
		}
		
	}
	void BackPropWeights() {
		
		if (HiddenLayerCount > 1) {
			for (int row = 0; row < OutputSize; row++) {
				for (int col = 0; col < HiddenLayerSize; col++) {
					OutputLayer.GetLayerWeights()[row][col] -= HiddenLayers.back().GetLayerActivatedNeurons()[col] * OutputLayer.GetLayerErrorDeltas()[row] * InitialLearningRate;
				}
			}
			for (int index = HiddenLayerCount - 2; index >= 1; index--) {
				for (int row = 0; row < HiddenLayerSize; row++) {
					for (int col = 0; col < HiddenLayerSize; col++) {
						HiddenLayers[index].GetLayerWeights()[row][col] -= HiddenLayers[index - 1].GetLayerActivatedNeurons()[col] * HiddenLayers[index].GetLayerErrorDeltas()[row] * InitialLearningRate;
					}
				}
			}
			for (int row = 0; row < HiddenLayerSize; row++) {
				for (int col = 0; col < HiddenLayerSize; col++) {
					HiddenLayers[0].GetLayerWeights()[row][col] -= FirstHiddenLayer.GetLayerActivatedNeurons()[col] * HiddenLayers[0].GetLayerErrorDeltas()[row] * InitialLearningRate;
				}
			}
			for (int row = 0; row < HiddenLayerSize; row++) {
				for (int col = 0; col < InputSize; col++) {
					FirstHiddenLayer.GetLayerWeights()[row][col] -= InputLayer.GetLayerActivatedNeurons()[col] * FirstHiddenLayer.GetLayerErrorDeltas()[row] * InitialLearningRate;
				}
			}

		}
		else {
			for (int row = 0; row < OutputSize; row++) {
				for (int col = 0; col < HiddenLayerSize; col++) {
					OutputLayer.GetLayerWeights()[row][col] -= FirstHiddenLayer.GetLayerActivatedNeurons()[col] * OutputLayer.GetLayerErrorDeltas()[row] * InitialLearningRate;
				}
			}
			for (int row = 0; row < HiddenLayerSize; row++) {
				for (int col = 0; col < InputSize; col++) {
					FirstHiddenLayer.GetLayerWeights()[row][col] -= InputLayer.GetLayerActivatedNeurons()[col] * FirstHiddenLayer.GetLayerErrorDeltas()[row] * InitialLearningRate;
				}
			}
		}
	}
	void BackPropBiases() {
		if (HiddenLayerCount > 1) {
			for (int num = 0; num < OutputSize; num++) {
				OutputLayer.GetLayerBiases()[num] -= OutputLayer.GetLayerErrorDeltas()[num] * InitialLearningRate;
			}
			for (int index = HiddenLayerCount - 2; index >= 0; index--) {
				for (int num = 0; num < HiddenLayerSize; num++) {
					HiddenLayers[index].GetLayerBiases()[num] -= HiddenLayers[index].GetLayerErrorDeltas()[num] * InitialLearningRate;
				}
			}
			for (int num = 0; num < OutputSize; num++) {
				FirstHiddenLayer.GetLayerBiases()[num] -= FirstHiddenLayer.GetLayerErrorDeltas()[num] * InitialLearningRate;
			}
		}
	}
	
public:
	NeuralNetwork(double InitialLearningRate, ActivationType type)
		:InitialLearningRate(InitialLearningRate)
	{
		if (HiddenLayerCount < 1) {
			throw HiddenLayerCountException("hidden layer count must be greater than 0");
		}
		HiddenLayers.reserve(HiddenLayerCount - 1);
		HiddenLayers.resize(HiddenLayerCount - 1);


	}
	NeuralNetwork(double InitialLearningRate, ActivationType type, const char* )
		:InitialLearningRate(InitialLearningRate)
	{
		if (HiddenLayerCount < 1) {
			throw HiddenLayerCountException("hidden layer count must be greater than 0");
		}
		HiddenLayers.reserve(HiddenLayerCount - 1);
		HiddenLayers.resize(HiddenLayerCount - 1);


	}
	~NeuralNetwork() {

	}

	void PropogateForward() {
		FirstHiddenLayer.CalcForwardProp(InputLayer);


		for (int index = 1; index < HiddenLayers.size();index++) {
			HiddenLayers[index].CalcForwardProp(HiddenLayers[index - 1]);
		}

		if (HiddenLayerCount == 1) {
			OutputLayer.CalcForwardProp(FirstHiddenLayer);
			return;
		}

		if (HiddenLayerCount == 2) {
			HiddenLayers[0].CalcForwardProp(FirstHiddenLayer);
		}
		OutputLayer.CalcForwardProp(HiddenLayers.back());
		if (CurrentTargetIndex == GetMaxValue(OutputLayer.GetLayerActivatedNeurons())) {
			count++;
		}
		
	}
	void PropogateBackwards() {
		BackPropActivations();
		BackPropWeights();
		BackPropBiases();
	}
	using CSV_FILE = const char*;
	void LoadTrainingData(CSV_FILE FileName, size_t RowNum) {
		std::ifstream TrainingData(FileName);
		
		char character;
		std::string number;
		std::vector<double> row;
		row.reserve(InputSize+1);
		ProcessingData.resize(RowNum);
		
		for (std::vector<double>& index : ProcessingData) {
			index.reserve(InputSize+1);
		}
		int index = 0;
		while (TrainingData.get(character) && index < RowNum) {
			if (character == ',') {
				row.push_back(std::stod(number));
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
	}
	void Train() {
		for (const std::vector<double>& row : ProcessingData) {
			CurrentTargetIndex = row[0];
			std::vector<double> temp = { row.begin() + 1, row.end() };
			InputLayer.SetLayerNeurons(temp);
			PropogateForward();
			PropogateBackwards();
		}
		
	}
	void PrintOutput() const {
		OutputLayer.print();
		std::cout << (float)count/10000 << std::endl;

	}
};