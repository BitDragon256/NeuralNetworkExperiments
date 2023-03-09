#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <random>

struct Matrix
{
	std::vector<std::vector<double>> data;
	size_t width, height;

	Matrix() :
		width{ 0 }, height{ 0 }
	{}
	Matrix(size_t width, size_t height) :
		width{ width }, height{ height }, data{ height, std::vector<double>( width ) }
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				data[i][j] = 0;
			}
		}
	}
	Matrix(std::vector<std::vector<double>> values) :
		width((values.size() > 0) ? values[0].size() : 0), height(values.size()), data{values}
	{}

	Matrix operator*(const Matrix& other)
	{
		if (width != other.height)
		{
			std::cout << "matrices can't be multiplied\n";
			return { 0, 0 };
		}

		Matrix result{ other.width, height };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < other.width; j++)
			{
				for (int k = 0; k < width; k++)
				{
					result.data[i][j] += data[i][k] * other.data[k][j];
				}
			}
		}

		return result;
	}
	Matrix& operator=(const Matrix& other)
	{
		data = other.data;
		height = data.size();
		if (height > 0)
			width = data[0].size();
		return *this;
	}

	void print()
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				std::cout << data[i][j] << " ";
			}
			std::cout << '\n';
		}
	}
	Matrix for_each(std::function<double(double)> fun)
	{
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				data[i][j] = fun(data[i][j]);
		return *this;
	}
};

Matrix make_vector(std::vector<double> values)
{
	Matrix res{ 1, values.size() };
	for (int i = 0; i < res.height; i++)
	{
		res.data[i][0] = values[i];
	}
	return res;
}
Matrix make_vector(size_t size)
{
	return { 1, size };
}
Matrix make_matrix(size_t width, size_t height)
{
	return { width, height };
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}
double activation(double in)
{
	return sigmoid(in);
}

class NeuralNet
{
public:
	NeuralNet(std::vector<size_t> layers) :
		neurons{ layers.size() }, weights{ layers.size() - 1 }
	{
		std::random_device rd{  };
		std::mt19937 gen{ rd() };

		for (int l = 0; l < layers.size(); l++)
		{
			neurons[l] = make_vector(layers[l]);
			if (l > 0)
			{
				auto& weightMatrix = weights[l - 1];
				weightMatrix = make_matrix(layers[l - 1], layers[l]);

				// randomizing weights based on normalized xavier weight initialization
				std::normal_distribution<> nd{ 0, sqrt(6) / sqrt(layers[l - 1] + layers[l]) };
				for (int i = 0; i < weightMatrix.height; i++)
				{
					for (int j = 0; j < weightMatrix.width; j++)
					{
						weightMatrix.data[i][j] = nd(gen);
					}
				}
			}
		}
	}

	void train_iter(size_t iterations)
	{

	}
	void train(std::vector<double> in, std::vector<double> out)
	{

	}
	std::vector<double> feed_forward(std::vector<double> input)
	{
		neurons[0] = make_vector(input);

		for (int l = 1; l < neurons.size(); l++)
		{
			auto& weightMatrix = weights[l - 1];
			auto& in = neurons[l - 1];
			auto& out = neurons[l];

			out = weightMatrix * in;
			out.for_each([](double in) { return activation(in); });
		}

		auto& outputNeurons = neurons[neurons.size() - 1];
		std::vector<double> output( outputNeurons.height );
		for (int i = 0; i < outputNeurons.height; i++)
		{
			output[i] = outputNeurons.data[i][0];
		}
		return output;
	}

private:
	std::vector<Matrix> weights;
	std::vector<Matrix> neurons;
};

