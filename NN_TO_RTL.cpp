﻿// NN_TO_RTL.cpp : Defines the entry point for the application.
//

#include "NN_TO_RTL.h"

using namespace std;
NeuralNetwork* NeuralNetwork::nn = nullptr;
istream& operator>>(istream& in, ACT_FUNC& x) {
	int val;
	if (in >> val) {
		switch (val)
		{
		case RELU: 
		case SIGMOID: 
		case TANH:
		case SOFTMAX:
		x = ACT_FUNC(val);
		break;
		default:
			throw out_of_range("Invalid value for type ACT FUNC");
			break;
		}
	}
}
Precptron::Precptron(ACT_FUNC func, int no_inputs, std::vector< float>& weights,Layer* l, float bias):
	_act_func(func),
	_no_inputs(no_inputs),
	owner(l),
	_bias(0) {
	for (auto w : weights) {
		_weights.push_back(w);
	}
}

Layer::Layer(ACT_FUNC func, Layer* previous_layer, int no_preceptron, int layer_no,IO_TYPE io_type):
	_act_func(func),
	_previous_layer(previous_layer),
	_no_preceptron(no_preceptron)
{
	std::vector<std::vector<float>> weight_mat = getWeights(layer_no, no_preceptron, io_type);
	std::vector<float> bias_vec = getBias(layer_no, no_preceptron,io_type);
	size_t  ninputs = 0;
	if (_previous_layer)
		ninputs = _previous_layer->_precptron.size();
	else
		ninputs = NeuralNetwork::getInstance(io_type)->getInputs();
	for (int i = 0; i < no_preceptron; i++) {
		Precptron* p = new Precptron(_act_func, ninputs, weight_mat[i], this, bias_vec[i]);
		_precptron.push_back(p);
	}
}
std::vector<float> Layer::getWeightsNode(IO_TYPE io_type,int layer,int precptron) {
	size_t  ninputs = 0;
	if (_previous_layer)
		ninputs = _previous_layer->_precptron.size();
	else
		ninputs = NeuralNetwork::getInstance(io_type)->getInputs();
	std::vector<float> weight;
	if (io_type == IO_TYPE::CONSOLE) {
		cout << "Get weights for " << layer << "-" << precptron;
		for (size_t i = 0; i < ninputs; i++) {
			float w;
			cin >> w;
			weight.push_back(w);
		}
	}
	return weight;
}

std::vector<std::vector<float>> Layer::getWeights(int layer_no, int no_preptron, IO_TYPE io_type) {
	std::vector<std::vector<float>> weight_mat;
	for (int i = 0; i < no_preptron; i++) {
		weight_mat.push_back(getWeightsNode(io_type, layer_no, i));
	}
	return weight_mat;
}
std::vector<float> Layer::getBias(int layer_no, int no_precptron, IO_TYPE io_type) {
	std::vector<float> biasVec;
	if (io_type == IO_TYPE::CONSOLE) {
		cout << "Get Bias vec for each precptron in Layer" << layer_no;
		for (int i = 0; i < no_precptron; i++) {
			float b;
			cin >> b;
			biasVec.push_back(b);
		}
	}
	return biasVec;
}
void NeuralNetwork::get_IO_number() 
{
	if (_io_type == IO_TYPE::CONSOLE) {
		cout << "Number of Layers";
		cin >> _no_layers;
		cout << "Number of inputs";
		cin >> _no_inputs;
		cout << "Number of outputs";
		cin >> _no_outputs;
	}
}

int NeuralNetwork::get_no_precptron(int i) {
	int n = 0;
	if (_io_type == CONSOLE) {
		cout << "No of precptron in layer " << i;
		cin >> n;
	}
	return n;
}
NeuralNetwork::NeuralNetwork(IO_TYPE io_type):
	_no_layers(0),
	_no_inputs(0),
	_no_outputs(0),
	_io_type(io_type)
{
	get_IO_number();
}

void NeuralNetwork::createNetwork() {
	for (int i = 0; i < _no_layers; i++) {
		Layer* prev;
		if (i == 0)
			prev = nullptr;
		else
			prev = network[i - 1];
		int no_precptron = get_no_precptron(i);

		ACT_FUNC func = Layer::getActFunc(i + 1);
		Layer* l = new Layer(func, prev, no_precptron, i + 1, _io_type);
		network.push_back(l);
		if (prev)
			prev->setNextLayer(l);
	}
}

ACT_FUNC Layer::getActFunc(int layer_no) {
	ACT_FUNC func;
	cout << "Activation functon for layer" << layer_no << "1:Relu 2:Sigmoid 3:TANH 4:SoftMax " << endl;
	cin >> func;
	return func;
}

NeuralNetwork* NeuralNetwork::getInstance(IO_TYPE io_type) {
	if (!nn) {
		nn = new NeuralNetwork(io_type);
	} 
	return nn;
}

int main()
{
	NeuralNetwork* network = NeuralNetwork::getInstance(IO_TYPE::CONSOLE);
	network->createNetwork();
	return 0;
}
