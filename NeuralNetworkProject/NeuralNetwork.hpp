#pragma once
#include <vector>
#include <array>
#include <exception>
#include <fstream>
#include <string>
#include <math.h>

#include "Matrix.hpp"
#include "Vector.hpp"
#include "NetworkLayer.hpp"
template<typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T> vector) {
	for (int index = 0; index < vector.size(); index++) {
		o << vector[index] << " ";
	}
	o << std::endl;
	return o;
}

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
int GetMaxValueIndex(const Vector<size>& vector) {
	int pos = 0;
	for (int index = 1; index < vector.GetSize(); index++) {
		if (vector[pos] < vector[index]) {
			pos = index;
		}
	}
	return pos;
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
	ActivationType type;
	void BackPropActivations() {
		Vector<OutputSize>(*OutputDerivativeActivationFunc)(const Vector<OutputSize>& vector);
		Vector<HiddenLayerSize>(*HiddenLayerDerivativeActivationFunc)(const Vector<HiddenLayerSize>& vector);
		switch (type) {
			case ActivationType::SIGMOID:
				OutputDerivativeActivationFunc = SigmoidifiedDerivative<OutputSize>;
				HiddenLayerDerivativeActivationFunc = SigmoidifiedDerivative<HiddenLayerSize>;
				break;
			case ActivationType::RELU:
				OutputDerivativeActivationFunc = ReluifiedDerivative<OutputSize>;
				HiddenLayerDerivativeActivationFunc = ReluifiedDerivative<HiddenLayerSize>;
				break;
			default:
				OutputDerivativeActivationFunc = SigmoidifiedDerivative<OutputSize>;
				HiddenLayerDerivativeActivationFunc = SigmoidifiedDerivative<HiddenLayerSize>;
		}

		// calculates error delta for output layer
		Vector<OutputSize> OutputLayerDerivatives = OutputDerivativeActivationFunc(OutputLayer.GetLayerNeurons());
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

		// calculates error deltas for rest of layers
		if (HiddenLayerCount > 1) {
			for (int index = HiddenLayerCount - 2; index >= 0; index--) {
				Vector<HiddenLayerSize> CurrentLayerDerivatives = HiddenLayerDerivativeActivationFunc(HiddenLayers[index].GetLayerNeurons());
				if (index == HiddenLayerCount - 2) {
					HiddenLayers[index].SetLayerErrorDeltas((OutputLayer.GetLayerWeights().GetTransposed() * OutputLayer.GetLayerErrorDeltas()) * CurrentLayerDerivatives);
				}
				else {
					HiddenLayers[index].SetLayerErrorDeltas((HiddenLayers[index + 1].GetLayerWeights().GetTransposed() * HiddenLayers[index + 1].GetLayerErrorDeltas()) * CurrentLayerDerivatives);
				}
			}
			Vector<HiddenLayerSize> CurrentLayerDerivatives = HiddenLayerDerivativeActivationFunc(FirstHiddenLayer.GetLayerNeurons());
			FirstHiddenLayer.SetLayerErrorDeltas((HiddenLayers[0].GetLayerWeights().GetTransposed() * HiddenLayers[0].GetLayerErrorDeltas()) * CurrentLayerDerivatives);
		}
		else {
			Vector<HiddenLayerSize> CurrentLayerDerivatives = HiddenLayerDerivativeActivationFunc(FirstHiddenLayer.GetLayerNeurons());
			//std::cout << OutputLayer.GetLayerWeights() << std::endl;
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
					FirstHiddenLayer.GetLayerWeights()[row][col] -= InputLayer.GetLayerNeurons()[col] * FirstHiddenLayer.GetLayerErrorDeltas()[row] * InitialLearningRate;
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
					FirstHiddenLayer.GetLayerWeights()[row][col] -= InputLayer.GetLayerNeurons()[col] * FirstHiddenLayer.GetLayerErrorDeltas()[row] * InitialLearningRate;
				}
			}
		}
	}
	void BackPropBiases() {
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
	
public:
	NeuralNetwork(double InitialLearningRate, ActivationType type)
		:InitialLearningRate(InitialLearningRate), CurrentTargetIndex(NULL), InputLayer(ActivationType::NONE), type(type), OutputLayer(ActivationType::SOFTMAX)
	{
		if (HiddenLayerCount < 1) {
			throw HiddenLayerCountException("hidden layer count must be greater than 0");
		}
		FirstHiddenLayer = Layer<HiddenLayerSize, InputSize>(type);
		for (int index = 0; index < HiddenLayerCount - 1; index++) {
			HiddenLayers.emplace_back(Layer<HiddenLayerSize, HiddenLayerSize>(type));
		}
		HiddenLayers.shrink_to_fit();
		

	}
	
	~NeuralNetwork() {

	}

	void PropogateForward() {
		FirstHiddenLayer.CalcForwardProp(InputLayer);

		if (HiddenLayerCount == 1) {
			OutputLayer.CalcForwardProp(FirstHiddenLayer);
			if (CurrentTargetIndex == GetMaxValueIndex(OutputLayer.GetLayerActivatedNeurons())) {
				count++;
			}
			return;
		}

		HiddenLayers[0].CalcForwardProp(FirstHiddenLayer);
		for (int index = 1; index < HiddenLayers.size();index++) {
			HiddenLayers[index].CalcForwardProp(HiddenLayers[index - 1]);
		}

		OutputLayer.CalcForwardProp(HiddenLayers.back());
		if (CurrentTargetIndex == GetMaxValueIndex(OutputLayer.GetLayerActivatedNeurons())) {
			count++;
		}
		
	}
	void PropogateBackwards() {
		BackPropActivations();
		BackPropWeights();
		BackPropBiases();
	}
	using CSV_FILE = const char*;
	bool LoadData(CSV_FILE FileName, size_t RowNum, const bool RemoveFirstLine) {
		std::ifstream TrainingData(FileName);
		if (!TrainingData) {
			std::cout << "file does not exist" << std::endl;
			return false;
		}
		ProcessingData.clear();
		char character;
		if (RemoveFirstLine) {
			std::string temp;
			TrainingData >> temp;
			TrainingData.get(character);
			RowNum--;
		}
		
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
				row.push_back(std::stod(number));
				number.clear();
				ProcessingData[index] = row;
				row.clear();
				index++;
			}
			else if (character >= 48 && character <= 57) {
				number += character;
			}
		}
		if (!TrainingData.get(character) && !number.empty()) {
			row.push_back(std::stod(number));
			ProcessingData[index] = row;
		}
		char f = NULL;
		ProcessingData.resize(index);
		ProcessingData.shrink_to_fit();
		TrainingData.close();
		
		return true;
	}
	void Train() {
		for (const std::vector<double>& row : ProcessingData) {
			CurrentTargetIndex = row[0];
			
			std::vector<double> temp = { row.begin()+1, row.end() };
			Vector<784> v = temp;
			v /= 255;
			InputLayer.SetLayerNeurons(v);
			PropogateForward();
			PropogateBackwards();
		}
	}

	void Run() {
		count = 0;
		for (const std::vector<double>& row : ProcessingData) {
			CurrentTargetIndex = row[0];
			std::vector<double> temp = { row.begin() + 1, row.end() };
			Vector<784> v = temp;
			v /= 255;
			InputLayer.SetLayerNeurons(v);
			PropogateForward();
		}
	}

	void PrintOutput() const {
		//OutputLayer.print();
		std::cout << "percentage correct: " << (double)count / ProcessingData.size() * 100 << std::endl;
		//std::cout << CurrentTargetIndex << std::endl;
		std::cout << std::endl;
		
	}
};