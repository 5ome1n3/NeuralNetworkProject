#pragma once
#include <vector>
#include <array>
#include <initializer_list>
#include "Vector.hpp"

template<size_t rows, size_t cols>
class Matrix {
private:
	std::vector<std::vector<double>> MatrixArray;
	void AdjustSize() {
		MatrixArray.reserve(rows);
		MatrixArray.resize(rows);
		for (std::vector<double>& col : MatrixArray) {
			col.reserve(cols);
			col.resize(cols);
		}
	}
public:
	Matrix() {
		AdjustSize();
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				MatrixArray[row][col] = -1 + (float)rand() / RAND_MAX;
			}
		}
	}
	Matrix(std::array<std::array<double, cols>, rows> matrix)
		:MatrixArray(MatrixArray)
	{
		AdjustSize();
	}
	Matrix(const Matrix<rows, cols>& matrix) {
		AdjustSize();
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				MatrixArray[row][col] = matrix[row][col];
			}
		}
	}
	Matrix(const std::initializer_list<std::initializer_list<double>>& matrix) {
		AdjustSize();
		int RowIndex = 0;
		for (const std::initializer_list<double>& row : matrix) {
			int ColIndex = 0;
			for (const double& col : row) {
				MatrixArray[RowIndex][ColIndex++] = col;
			}
			RowIndex++;
		}
		
		
	}

	const Matrix<cols, rows> GetTransposed() const {
		Matrix<cols, rows> matrix;
		for (int row = 0; rows < rows; row++) {
			for (int col = 0; col < cols; col++) {
				matrix[col][row] = MatrixArray[row][col];
			}
		}
		return matrix;
	}

	int GetRowLength() const {
		return rows;
	}
	int GetColumnLength() const {
		return cols;
	}

	const Matrix<rows, cols>& operator=(Matrix<rows, cols> matrix) {
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				MatrixArray[row][col] = matrix[row][col];
			}
		}
		return *this;
	}
	std::vector<double>& operator[](int index) {
		return MatrixArray[index];
	}
	const std::vector<double>& operator[](int index) const {
		return MatrixArray[index];
	}

	template<int NewCols>
	Matrix<rows, NewCols> operator*(Matrix<cols, NewCols> matrix) const {
		Matrix<rows, NewCols> result;
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < NewCols; col++) {
				for (int index = 0; index < cols; index++) {
					result[row][col] += MatrixArray[row][index] * matrix[index][col];
				}
			}
		}
		return result;
	}

	Matrix<rows, cols> operator+(const Matrix<cols, rows>& matrix) const {
		Matrix<rows, cols> result;
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = MatrixArray[row][col] + matrix[row][col];
			}
		}
		return result;
	}
	Matrix<rows, cols> operator-(const Matrix<cols, rows>& matrix) const {
		Matrix<rows, cols> result;
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = MatrixArray[row][col] - matrix[row][col];
			}
		}
		return result;
	}
	Vector<rows> operator*(const Vector<cols>& vector) const {
		Vector<rows> result;
		for (int row = 0; row < rows; row++) {
			for (int index = 0; index < cols; index++) {
				result[row] += MatrixArray[row][index] * vector[index];
			}
		}
		return result;
	}
};


template<size_t rows, size_t cols >
std::ostream& operator<<(std::ostream& output, const Matrix<rows, cols>& m) {
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			output << m[row][col] << " ";
		}
		output << "\n";
	}
	return output;
}