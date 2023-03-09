#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
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
	Matrix transposed()
	{
		Matrix res { height, width };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				res.data[j][i] = data[i][j];
			}
		}
		return res;
	}

	void print()
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				std::cout << std::setw(6) << std::setprecision(3) << data[i][j] << " ";
			}
			std::cout << '\n';
		}
		std::cout << '\n';
	}
	Matrix for_each(std::function<double(double)> fun)
	{
		Matrix res { width, height };
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				res.data[i][j] = fun(data[i][j]);
		return res;
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
double sigmoid_derivative(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}
double activation(double in)
{
	return sigmoid(in);
}
double activation_derivative(double in)
{
	return sigmoid_derivative(in);
}
double error_squared(double x)
{
	return x*x;
}
double error_squared_derivative(double x)
{
	return 0.5 * x;
}
double loss(double in)
{
	return error_squared(in);
}
double loss_derivative(double in)
{
	return error_squared_derivative(in);
}


class NeuralNet
{
public:
	NeuralNet(std::vector<size_t> layers, double learningRate) :
		neurons{ layers.size() }, weights{ layers.size() - 1 }, learningRate{ learningRate }
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
		auto outMat = make_vector(out);
		
		auto predict = feed_forward(in);
		
		std::vector<Matrix> deltas( neurons.size() );
		
		deltas[deltas.size() - 1] = make_vector(outMat.height);
		for (int i = 0; i < deltas[deltas.size() - 1].height; i++)
		{
			//std::cout << predict[i] << " " << outMat.data[i][0] << " " << predict[i] - outMat.data[i][0] << '\n';
			//deltas[deltas.size() - 1].data[i][0] = loss_derivative(outMat.data[i][0] - predict[i]);
			deltas[deltas.size() - 1].data[i][0] = (predict[i] - out[i]) * predict[i] * (1 - predict[i]);
		}
		
		//deltas[deltas.size() - 1].print();
		
		for (int l = neurons.size() - 2; l >= 0 ; l--)
		{
			auto& weightMatrix = weights[l];
			deltas[l] = make_vector(neurons[l].data.size());
			//deltas[l] = weightMatrix.transposed().for_each(sigmoid_derivative) * deltas[l + 1];
			for (int i = 0; i < neurons[l].data.size(); i++)
			{
				deltas[l].data[i][0] = 0;
				for (int j = 0; j < neurons[l + 1].data.size(); j++)
				{
					deltas[l].data[i][0] += weightMatrix.data[j][i] * deltas[l + 1].data[j][0];
				}
				deltas[l].data[i][0] *= neurons[l].data[i][0] * (1 - neurons[l].data[i][0]);
			}
			
		}
		
		for (int l = 0; l < weights.size(); l++)
		{
			auto& weightMatrix = weights[l];
			for (int i = 0; i < weightMatrix.height; i++)
			{
				for (int j = 0; j < weightMatrix.width; j++)
				{
					double deltaWeight = -learningRate * neurons[l + 1].data[i][0] * deltas[l].data[j][0];
					weightMatrix.data[i][j] += deltaWeight;
				}
			}
		}
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
			out = out.for_each(activation);
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
	double learningRate;
};

