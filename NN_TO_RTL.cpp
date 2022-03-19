// NN_TO_RTL.cpp : Defines the entry point for the application.
//

#include "NN_TO_RTL.h"
#include <assert.h>
#include<string>  
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

float Precptron::getWeight(int i) {
	return _weights[i];
}

float Precptron::getBias() {
	return _bias;
}
ACT_FUNC Precptron::getActFunc() {
	return _act_func;
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

std::vector<Layer*>& NeuralNetwork::getLayers() {
	return network;
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
/*/std::unordered_map<Name*, Port*> Netlist::inports;
std::unordered_map<Name*, Port*>  Netlist::outports;
std::unordered_map<Name*, Net*>  Netlist::nets;
std::unordered_map<Name*, Instance*>  Netlist::insts;*/
Netlist::Netlist(NeuralNetwork* nt) {
	createInports(nt->getInputs());
	createOutports(nt->getOutputs());
	std::vector<Layer*> lvec = nt->getLayers();
	std::vector<Net*> previous_layer_nets;
	for (int i = 0; i < lvec.size(); i++) {
		previous_layer_nets = createNetlistForLayer(lvec[i], (i == 0), (i == lvec.size() - 1), previous_layer_nets);
	}
}

std::vector<Net*> Netlist::createNetlistForLayer(Layer* l, bool is_input_layer, bool is_output_layer, std::vector<Net*>& previous_layer_nets){
	std::vector<Net*> outputNets;
	int in = previous_layer_nets.size();
	int out = l->getNoOfOutputs();;
	if (is_input_layer) {
		for (auto port : inports) {
			previous_layer_nets.push_back(port.second->np);
		}
	}
	if (!is_output_layer) {
		for(int i =0 ;i<out;i++)
			outputNets.push_back(createNet());
	}
	else {
		for (auto port : outports) {
			outputNets.push_back(port.second->np);
		}
	}
	for (int i = 0; i < out; i++) {
		Precptron* p = l->getPreceptron(i);
		ProcessPreceptron(p, previous_layer_nets, outputNets[i]);
	}
	return outputNets;
}

Net* Netlist::createAdd(Net* i1, Net* i2) {
	Net* onp = createNet();
	Instance* ip = new Instance(InstType::add);
	insts[ip->n] = ip;
	NhookPin(ip->input_pins[0], i1);
	NhookPin(ip->input_pins[1], i2);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::createMul(Net* np, float val) {
	Net* onp = createNet();
	Net* cnp = createNet(true, val);
	Instance* ip = new Instance(InstType::mult);
	insts[ip->n] = ip;
	NhookPin(ip->input_pins[0], np);
	NhookPin(ip->input_pins[1], cnp);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::addBias(Net* np, float val) {
	Net* onp = createNet();
	Net* cnp = createNet(true, val);
	Instance* ip = new Instance(InstType::add);
	insts[ip->n] = ip;
	NhookPin(ip->input_pins[0], np);
	NhookPin(ip->input_pins[1], cnp);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::createActFunc(Net* np, Net* onp,ACT_FUNC f) {
	if(!onp)
		onp = createNet();
	Instance* ip = new Instance(f);
	insts[ip->n] = ip;
	NhookPin(ip->input_pins[0], np);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::createNet(bool isConst, float val) {
	Name* n = Name::getUniqueName(NType::Net);
	Net* np = NULL;
	if (isConst) {
		np = Net::createConstNet(n, val);
	}
	else {
		np = new Net(n);
	}
	nets[np->n] = np;
	return np;
}
void Netlist::createInports(int i) {
	for (int j = 0; j < i; j++) {
		Port* p = new Port(this, Dir::in);
		inports[p->getName()] = p;
	}
}

void Netlist::createOutports(int i) {
	for (int j = 0; j < i; j++) {
		Port* p = new Port(this, Dir::out);
		outports[p->getName()] = p;
	}
}
void Netlist::ProcessPreceptron(Precptron* p, std::vector<Net*>& inputNets, Net* onp) {
	std::vector<Net*> mulOut;
	for (int i = 0; i < inputNets.size(); i++) {
		mulOut.push_back(createMul(inputNets[i], p->getWeight(i)));
	}
	Net* snp = mulOut[0];
	for (int i = 1; i < inputNets.size(); i++) {
		snp = createAdd(snp, mulOut[i]);
	}
	snp = addBias(snp, p->getBias());
	createActFunc(snp, onp, p->getActFunc());
}
Port::Port(Netlist* nl, Dir dir) {
	this->nl = nl;
	this->dir = dir;
	n = Name::getUniqueName(NType::Port);
	pin =new Pin(dir, true, n);
	np = new Net(n, true);
	NhookPin(pin, np);
	nl->nets[n] = np;

}
Name* Port::getName() {
	return n;
}

std::map< float, Net*> Net::constMap;
Net::Net(Name* n, bool isPort, bool isConst, float val) {
	this->n = n;
	this->isPort = isPort;
	this->pin = NULL;
	this->isConst = isConst;
	this->val = val;
}

Net* Net::createConstNet(Name* n,float  val) {
	if (constMap.find(val) == constMap.end()) {
		constMap[val] = new Net(n, false, true, val);
	}
	return constMap[val];
}
Instance::Instance(ACT_FUNC f) {
	input_pins.push_back(new Pin(Dir::in, false, Name::getNameForStr("I")));
	output_pin = new Pin(Dir::out, false, Name::getNameForStr("O"));
}
Instance::Instance(InstType t) {
	n = Name::getUniqueName(NType::Instance);
	type = t;
	switch (t) {
	case InstType::add:
		createAdder();
		break;
	case InstType::mult:
		createMult();
		break;
	case InstType::reg:
		createReg();
		break;
	}
}

void Instance::createAdder() {
	for (int i = 0; i < 2; i++) {
		input_pins.push_back(new Pin(Dir::in, false, Name::getNameForStr((i==0)?"A":"B")));
	}
	output_pin = new Pin(Dir::out, false, Name::getNameForStr("O"));
}

void Instance::createMult() {
	for (int i = 0; i < 2; i++) {
		input_pins.push_back(new Pin(Dir::in, false, Name::getNameForStr((i == 0) ? "A" : "B")));
	}
	output_pin = new Pin(Dir::out, false, Name::getNameForStr("O"));
}

void Instance::createReg() {
	for (int i = 0; i < 2; i++) {
		input_pins.push_back(new Pin(Dir::in, false, Name::getNameForStr((i == 0) ? "D" : "clk")));
	}
	output_pin = new Pin(Dir::out, false, Name::getNameForStr("Q"));
}
Pin::Pin(Dir dir, bool istopIo, Name * n) {
	this->dir = dir;
	this->istopIO = istopIo;
	this->n = n;
	this->np = NULL;
	this->next = NULL;
}
void NhookPin(Pin* p, Net* n) {
	if (!n->pin) {
		n->pin = p;
	}
	else {
		Pin* t = n->pin;
		while (t->next) {
			t = t->next;
		}
		t->next = p;
	}
	p->np = n;
}

Name::Name(string s) {
	str = s;
}
int Name::port_c = 1;
int Name::net_c = 1;
int Name::instance_c = 1;
int Name::pin_c = 1;
unordered_map<string, Name*>  Name::map;
Name* Name::getNameForStr(string s) {
	if (map.find(s) != map.end()) {
		return map[s];
	}
	Name* n = new Name(s);
	map[s] = n;
	return n;
}
string Name::getNameStr(Name* n) {
	return n->str;
}
Name* Name::getUniqueName(NType t) {
	string s;
	switch (t) {
	case NType::Port:
		s = "port_" + to_string(port_c++);
		return getNameForStr(s);
	case NType::Net:
		s = "net_" + to_string(net_c++);
		return getNameForStr(s);
	case NType::Instance:
		s = "inst_" + to_string(instance_c++);
		return getNameForStr(s);
	case NType::Pin:
		s = "pin_" + to_string(pin_c++);
		return getNameForStr(s);
	}
}
int main()
{
	NeuralNetwork* network = NeuralNetwork::getInstance(IO_TYPE::CONSOLE);
	network->createNetwork();
	Netlist* nl = new Netlist(network);
	return 0;
}
