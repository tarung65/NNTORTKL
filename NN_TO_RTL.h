// NN_TO_RTL.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <unordered_map>;
enum ACT_FUNC
{
	RELU = 1,
	SIGMOID,
	TANH,
	SOFTMAX
};

enum IO_TYPE {
	CONSOLE = 1,
	File
};
class Layer;
class Precptron {
	ACT_FUNC _act_func;
	int _no_inputs;
	std::vector <float> _weights;
	float _bias;
	Layer* owner;
public:
	Precptron(ACT_FUNC func, int no_inputs, std::vector< float>& weights,Layer* l, float bias);
};


class Layer {
	ACT_FUNC _act_func;
	Layer* _previous_layer;
	Layer* _next_layer;
	int _no_preceptron;
	std::vector< Precptron*> _precptron;
	std::vector<float> getWeightsNode(IO_TYPE io_type, int layer, int precptron);
public:
	Layer(ACT_FUNC func, Layer* previous_layer, int no_preceptron, int layer_no, IO_TYPE io_type);
	std::vector<std::vector<float>> getWeights(int layer_no,int no_preptron, IO_TYPE io_type);
	std::vector<float> getBias(int layer_no,int no_preptron, IO_TYPE io_type);
	static ACT_FUNC getActFunc(int layer_no);
	void setNextLayer(Layer* next_layer) {
		this->_next_layer = next_layer;
	}
};

class NeuralNetwork {
	static NeuralNetwork* nn;
	int _no_layers;
	int _no_inputs;
	int _no_outputs;
	IO_TYPE _io_type;
	std::vector<Layer*> network;
	void get_IO_number();
	int get_no_precptron(int i);
	NeuralNetwork(IO_TYPE io_type);
public:
	static NeuralNetwork* getInstance(IO_TYPE io_type);
	void createNetwork();
	int getInputs() {
		return _no_inputs;
	}
	int getOutputs() {
		return _no_outputs;
	}
};
class Port;
class Net;
class Instance;
class Pin;
enum class NType {
	Port, Net, Instance, Pin
};
class Name {
	static int port_c;
	static int net_c;
	static int pin_c;
	static int instance_c;
	std::string getUniqueName(NType t);
};
class Netlist {
	std::unordered_map<std::string, Port> ports;
	std::unordered_map<std::string, Net> ports;
	std::unordered_map<std::string, Instance> ports;
};

class Instance {
	std::string name;
	std::vector<Pin> input_pins;
	std::vector<Pin> output_pins;
};
class Net {
	std::string name;
	std::vector<Pin> pins;
};
// TODO: Reference additional headers your program requires here.
