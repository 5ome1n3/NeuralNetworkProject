#pragma once
#include "Vector.hpp"
#include "Matrix.hpp"



template<size_t size>
void Sigmoidify(Vector<size>& vector) {
	for (int index = 0; index < size; index++) {
		vector[index] = 1 / (1 + exp(-vector[index]));
	}
}


enum ActivationType {
	RELU,
	SOFTMAX,
	SIGMOID,
	NONE
};

template<size_t CurLayerSize, size_t PrevLayerSize>
class Layer {
private:
	Vector<CurLayerSize> LayerNeurons;
	Vector<CurLayerSize> LayerActivatedNeurons;
	Vector<CurLayerSize> LayerBiases;
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
	const Vector<CurLayerSize>& GetLayerBiases() const {
		return LayerBiases;
	}
	const Matrix<CurLayerSize, PrevLayerSize>& GetLayerWeights() const {
		return LayerWeights;
	}
	const ActivationType& GetActivationType() const {
		return type;
	}
	const Vector<CurLayerSize>& GetLayerActivatedNeurons() const {
		return LayerActivatedNeurons;
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





