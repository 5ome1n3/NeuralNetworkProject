#pragma once
#include "Vector.hpp"
#include "Matrix.hpp"

double Sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

template<size_t size>
void Sigmoidify(Vector<size>& vector) {
	for (int index = 0; index < size; index++) {
		vector[index] = Sigmoid(vector[index]);
	}
}

template<size_t size>
Vector<size> SigmoidifiedDerivative(const Vector<size>& vector) {
	Vector<size> result;
	for (int index = 0; index < size; index++) {
		result[index] = Sigmoid(vector[index]) * (1 - Sigmoid(vector[index]));
	}
	return result;
}

double Relu(double x) {
	return x > 0 ? x : 0;
}

template<size_t size>
void Reluify(Vector<size>& vector) {
	for (int index = 0; index < size; index++) {
		vector[index] = Relu(vector[index]);
	}
}

template<size_t size>
Vector<size> ReluifiedDerivative(const Vector<size>& vector) {
	Vector<size> result;
	for (int index = 0; index < size; index++) {
		result[index] = vector[index] > 0 ? 1 : 0;
	}
	return result;
}

template<size_t size>
void Softmaxify(Vector<size>& vector) {
	double sum = 0;
	for (int index = 0; index < size; index++) {
		sum += exp(vector[index]);
	}
	for (int index = 0; index < size; index++) {
		vector[index] = exp(vector[index]) / sum;
	}
}

enum ActivationType {
	NONE,
	RELU,
	SOFTMAX,
	SIGMOID
};

template<size_t CurLayerSize, size_t PrevLayerSize>
class Layer {
private:
	Vector<CurLayerSize> LayerNeurons;
	Vector<CurLayerSize> LayerActivatedNeurons;
	Vector<CurLayerSize> LayerBiases;
	Vector<CurLayerSize> LayerErrorDeltas;
	Matrix<CurLayerSize, PrevLayerSize> LayerWeights;
	ActivationType type;
	void (*ApplyActivation)(Vector<CurLayerSize>& vector);

private:
	void SetActivationFunction() {
		switch (type) {
			case ActivationType::SIGMOID:
				ApplyActivation = Sigmoidify;
				break;
			case ActivationType::RELU:
				ApplyActivation = Reluify;
				break;
			case ActivationType::SOFTMAX:
				ApplyActivation = Softmaxify;
				break;
			case ActivationType::NONE:
				ApplyActivation = nullptr;
				break;
			default:
				ApplyActivation = Sigmoidify;
		}
	}
	void SetActivatedNeurons() {
		LayerActivatedNeurons = LayerNeurons;
		if (ApplyActivation != nullptr) {
			ApplyActivation(LayerActivatedNeurons);
		}
	}
public:
	Layer(ActivationType type)
		:type(type)
	{
		SetActivationFunction();
		SetActivatedNeurons();

		for (int row = 0; row < LayerWeights.GetRowLength(); row++) {
			for (int col = 0; col < LayerWeights.GetColumnLength(); col++) {
				double scale = 1;
				if (type == SIGMOID) {
					scale = std::sqrt(3.0 / LayerWeights.GetColumnLength());
				}
				else if (type == RELU) {
					scale = std::sqrt(2.0 / LayerWeights.GetColumnLength());
				}
				LayerWeights[row][col] = -1 + ((double)rand() / RAND_MAX)*2.0;
				LayerWeights[row][col] *= scale;
			}
			LayerBiases[row] = 0;
		}
	}
	
	Layer()
		:type(ActivationType::SIGMOID)
	{
		SetActivationFunction();
		SetActivatedNeurons();

		for (int row = 0; row < LayerWeights.GetRowLength(); row++) {
			for (int col = 0; col < LayerWeights.GetColumnLength(); col++) {
				//LayerWeights[row][col] = -1 + (((float)rand() / RAND_MAX) * 2);
			}
			//LayerBiases[row] = 0;
		}
	}

	const Vector<CurLayerSize>& GetLayerNeurons() const {
		return LayerNeurons;
	}
	const Matrix<CurLayerSize, PrevLayerSize>& GetLayerWeights() const {
		return LayerWeights;
	}
	Matrix<CurLayerSize, PrevLayerSize>& GetLayerWeights() {
		return LayerWeights;
	}
	const Vector<CurLayerSize>& GetLayerBiases() const {
		return LayerBiases;
	}
	Vector<CurLayerSize>& GetLayerBiases() {
		return LayerBiases;
	}
	const Vector<CurLayerSize>& GetLayerErrorDeltas() const {
		return LayerErrorDeltas;
	}
	const ActivationType& GetActivationType() const {
		return type;
	}
	const Vector<CurLayerSize>& GetLayerActivatedNeurons() const {
		return LayerActivatedNeurons;
	}
	const int& GetSize() const {
		return CurLayerSize;
	}

	void SetLayerNeurons(const Vector<CurLayerSize>& neurons) {
		LayerNeurons = neurons;
		SetActivatedNeurons();
	}
	void SetLayerWeights(const Matrix <CurLayerSize, PrevLayerSize>& weights) {
		LayerWeights = weights;
	}
	void SetLayerBiases(const Vector<CurLayerSize>& biases) {
		LayerBiases = biases;
	}
	void SetLayerErrorDeltas(const Vector<CurLayerSize>& ErrorDeltas) {
		LayerErrorDeltas = ErrorDeltas;
	}

	template<size_t PrevSize, size_t PrevPrevSize>
	void CalcForwardProp(const Layer<PrevSize, PrevPrevSize>& PrevLayer) {
		LayerNeurons = (LayerWeights * PrevLayer.GetLayerActivatedNeurons()) + LayerBiases;
		//std::cout << PrevLayer.GetLayerActivatedNeurons() << std::endl;
		SetActivatedNeurons();
	}
	void print() const {
		std::cout << LayerActivatedNeurons << std::endl;
	}
};





