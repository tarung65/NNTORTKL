// NN_TO_RTL.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <map>
using namespace std;
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
	float getWeight(int i);
	float getBias();
	ACT_FUNC getActFunc();
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
	int getNoOfOutputs() {
		return _no_preceptron;
	}
	Precptron* getPreceptron(int i) {
		return _precptron[i];
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
	std::vector<Layer*>& getLayers();
};
class Port;
class Net;
class Instance;
class Pin;
enum class NType {
	Port, Net, Instance, Pin
};
class Name {
	std::string str;
	static unordered_map<string, Name*> map;
	static int port_c;
	static int net_c;
	static int pin_c;
	static int instance_c;
public:
	Name(string s);
	static Name* getUniqueName(NType t);
	static std::string getNameStr(Name* n);
	static Name* getNameForStr(string s);
};
class Netlist {
public:
	std::unordered_map<Name*, Port*> inports;
	std::unordered_map<Name*, Port*> outports;
	std::unordered_map<Name*, Net*> nets;
	std::unordered_map<Name*, Instance*> insts;
	Netlist(NeuralNetwork* nt);
	void createInports(int i);
	void createOutports(int i);
	std::vector<Net*> createNetlistForLayer(Layer* l, bool is_input_layer, bool is_output_layer, std::vector<Net*>& previous_layer_nets);
	Net* createNet(bool isConst = false, float val = 0);
	Net* createAdd(Net* i1, Net* i2);
	Net* createMul(Net* np, float val);
	Net* addBias(Net* np, float val);
	Net* createActFunc(Net* np, Net* onp,ACT_FUNC f);
	void ProcessPreceptron(Precptron* p,std::vector<Net*>& inputNets,Net* onp);
};
enum class InstType {
	add,mult,reg,actFunc
};

class Instance {
public :
	Name* n;
	InstType type;
	ACT_FUNC func;
	std::vector<Pin*> input_pins;
	Pin* output_pin;
	Instance(InstType t);
	Instance(ACT_FUNC f);
	void createAdder();
	void createMult();
	void createReg();

};
class Net {
public:
	static std::map< float ,Net* > constMap;
	Name* n;
	bool isPort;
	Pin* pin;
	bool isConst;
	float val;
	Net(Name* n, bool isPort = false,bool isConst = false,float val =0);
	static Net* createConstNet(Name* n,float val);
};
enum class Dir {
	in, out
};
class Port {
public:
	Dir dir;
	Name* n;
	Net* np;
	Pin* pin;
	Netlist* nl;
	Port(Netlist* nl, Dir dir);
	Name* getName();
};
class Pin {
public:
	Dir dir;
	bool istopIO;
	Name* n;
	Net* np;
	Pin(Dir dir, bool istopIo, Name* n);
	Pin* next;
};
void NhookPin(Pin* p, Net* n);
// TODO: Reference additional headers your program requires here.
