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
Vector<size> SigmoidifiedDerivative(const Vector<size> vector) {
	Vector<size> result;
	for (int index = 0; index < vector.GetSize(); index++) {
		result[index] = Sigmoid(result[index]) * (1 - Sigmoid(result[index]));
	}
	return result;
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
			case ActivationType::NONE:
				ApplyActivation = nullptr;
				break;

		}
	}
	void SetActivatedNeurons() {
		LayerActivatedNeurons = LayerNeurons;
		if (ApplyActivation != nullptr)
			ApplyActivation(LayerActivatedNeurons);
	}
public:
	Layer(ActivationType type)
		:type(type)
	{
		SetActivationFunction();
		SetActivatedNeurons();
	}
	
	Layer()
		:type(ActivationType::NONE)
	{
		SetActivationFunction();
		SetActivatedNeurons();
	}

	const Vector<CurLayerSize>& GetLayerNeurons() const {
		return LayerNeurons;
	}
	const Matrix<CurLayerSize, PrevLayerSize>& GetLayerWeights() const {
		return LayerWeights;
	}
	const Vector<CurLayerSize>& GetLayerBiases() const {
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
		SetActivatedNeurons();
	}
	void print() {
		std::cout << LayerNeurons << std::endl;
		std::cout << LayerWeights << std::endl;
	}
};





