#pragma once
#include <vector>
#include <initializer_list>
#include <array>

template<size_t size>
class Vector {
private:
	std::vector<double> VectorArray;
public:
	Vector() {
		VectorArray.reserve(size);
		VectorArray.resize(size);
		for (int index = 0; index < size; index++) {
			VectorArray[index] = 0;
		}
	}
	Vector(const std::array<double, size>& array) {
		VectorArray.reserve(size);
		VectorArray.resize(size);
		for (int index = 0; index < VectorArray.size(); index++) {
			VectorArray[index] = array[index];
		}
	}
	Vector(const Vector<size>& vector) {
		VectorArray.reserve(size);
		VectorArray.resize(size);
		for (int index = 0; index < vector.GetSize(); index++) {
			VectorArray[index] = vector[index];
		}
	}
	Vector(std::initializer_list<double> vector) {
		VectorArray.reserve(size);
		VectorArray.resize(size);
		int index = 0;
		for (const double& val : vector) {
			VectorArray[index++] = val;
		}
	}
	~Vector() {

	}
	const Vector<size>& operator=(const Vector<size>& vector) {
		for (int index = 0; index < VectorArray.size(); index++) {
			VectorArray[index] = vector[index];
		}
		return *this;
	}

	int GetSize() const {
		return size;
	}

	double& operator[](int index) {
		return VectorArray[index];
	}
	const double& operator[](int index) const {
		return VectorArray[index];
	}

	Vector<size> operator+(const Vector<size>& vector) const {
		Vector<size> temp;
		for (int index = 0; index < size; index++) {
			temp[index] = VectorArray[index] + vector[index];
		}
		return temp;
	}
	Vector<size> operator*(const Vector<size>& vector) const {
		Vector<size> temp;
		for (int index = 0; index < size; index++) {
			temp[index] = VectorArray[index] * vector[index];
		}
		return temp;
	}
	Vector<size> operator/(const Vector<size>& vector) const  {
		Vector<size> temp;
		for (int index = 0; index < size; index++) {
			temp[index] = VectorArray[index] / vector[index];
		}
		return temp;
	}
	Vector<size> operator*(const int value) const {
		Vector<size> temp;
		for (int index = 0; index < size; index++) {
			temp[index] *= value;
		}
		return temp;
	}
	
	const Vector<size>& operator*=(const Vector<size>& vector) {
		for (int index = 0; index < size; index++) {
			VectorArray[index] *= vector[index];
		}
		return *this;
	}
	const Vector<size>& operator/=(const Vector<size>& vector) {
		for (int index = 0; index < size; index++) {
			VectorArray[index] /= vector[index];
		}
		return *this;
	}
	const Vector<size>& operator+=(const Vector<size>& vector) {
		for (int index = 0; index < size; index++) {
			VectorArray[index] += vector[index];
		}
		return *this;
	}
	const Vector<size>& operator-=(const Vector<size>& vector) {
		for (int index = 0; index < size; index++) {
			VectorArray[index] -= vector[index];
		}
		return *this;
	}
};

template<size_t size>
std::ostream& operator<<(std::ostream& output, const Vector<size>& v) {
	for (int index = 0; index < size;index++) {
		output << v[index] << " ";
	}
	return output;
}