// NN_TO_RTL.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include<unordered_set>
#include <map>
#include<fstream>
using namespace std;
enum ACT_FUNC
{
	RELU = 1,
	SIGMOID,
	TANH,
	SOFTMAX
};

class IFile {
private:
	static ifstream *f;
public:
	static bool isOpen();
	static void Open(string file);
	static string getline();
	static vector<float> getinputsVec();
	static void setPtr(ifstream* ptr);
	static void close();
};

class OFile {
private:
	static ofstream* f;
public:
	static bool isOpen();
	static bool open(string file);
	static ofstream& getStream();
	static void close();
};
enum IO_TYPE {
	CONSOLE = 1,
	File
};
class Layer;
class Precptron {
	ACT_FUNC _act_func;
	size_t _no_inputs;
	std::vector <float> _weights;
	float _bias;
	Layer* owner;
public:
	Precptron(ACT_FUNC func, size_t no_inputs, std::vector< float>& weights,Layer* l, float bias);
	float getWeight(size_t i);
	float getBias();
	ACT_FUNC getActFunc();
};


class Layer {
	ACT_FUNC _act_func;
	static const  string relu;
	static const  string sigmoid;
	static const  string tanh;
	static const  string softmax;
	IO_TYPE _io_type;
	Layer* _previous_layer;
	Layer* _next_layer;
	size_t _no_preceptron;
	std::vector< Precptron*> _precptron;
	std::vector<float> getWeightsNode(IO_TYPE io_type, int layer, size_t precptron);
public:
	Layer(ACT_FUNC func, Layer* previous_layer, size_t no_preceptron, int layer_no, IO_TYPE io_type);
	std::vector<std::vector<float>> getWeights(int layer_no,size_t no_preptron, IO_TYPE io_type);
	std::vector<float> getBias(int layer_no,size_t no_preptron, IO_TYPE io_type);
	static ACT_FUNC getActFunc(int layer_no, IO_TYPE _io_type);
	void setNextLayer(Layer* next_layer) {
		this->_next_layer = next_layer;
	}
	size_t getNoOfOutputs() {
		return _no_preceptron;
	}
	Precptron* getPreceptron(size_t i) {
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
	std::vector< Instance*> insts;
	std::vector<Net*> enable_net_vec;
	std::unordered_set<size_t> andInst;
	Net* clk;
	Netlist(NeuralNetwork* nt);
	void createInports(int i);
	void createOutports(int i);
	void ProcessPreceptron(Precptron* p, std::vector<Net*>& inputNets, Net* onp, Net* n_enableNet, Net* p_enableNet);
	std::vector<Net*> createNetlistForLayer(Layer* l, bool is_input_layer, bool is_output_layer, std::vector<Net*>& previous_layer_nets);
	Net* createNet(bool isConst = false, float val = 0);
	Net* createNetSingleBit();
	Net* createAdd(Net* i1, Net* i2);
	Net* createMul(Net* np, float val);
	Net* createAnd(vector<Net*>& inputs, Net* outNp);
	Net* addBias(Net* np, float val);
	Net* createActFunc(Net* np, Net* onp, ACT_FUNC f, Net* n_enableNet, Net* p_enableNet);
};
enum class InstType {
	add,mult,reg,actFunc,And
};

class Instance {
public :
	Name* n;
	InstType type;
	ACT_FUNC func;
	std::vector<Pin*> input_pins;
	std::vector<Pin*> output_pins;
	Instance(InstType t);
	Instance(int i);
	Instance(ACT_FUNC f);
	void createAdder();
	void createMult();
	void createReg();
	void createAnd(int i);

};
class Net {
public:
	static std::map< float ,Net* > constMap;
	Name* n;
	bool isPort;
	Pin* pin;
	bool isConst;
	float val;
	bool isSinglebit;
	Net(Name* n, bool isPort = false,bool isConst = false,float val =0, bool isSinglebit=false);
	static Net* createConstNet(Name* n,float val);
	static Net* createSingleBitNet(Name* n);
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
	Port(Netlist* nl, Dir dir, Name* n1 = NULL, bool is_Single=false);
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
class NetListWriter {
	ofstream& ofs;
	Netlist* nl;
public:
	NetListWriter(Netlist* nl);
	void writePort();
	void writeNets();
	void writeInst();
	void assignConst();
	void writeAndMod();
	static std::string getConstInBinary(float val);
};
// TODO: Reference additional headers your program requires here.
