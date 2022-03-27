// NN_TO_RTL.cpp : Defines the entry point for the application.
//

#include "NN_TO_RTL.h"
#include <assert.h>
#include<string> 
#include <sstream>
using namespace std;
NeuralNetwork* NeuralNetwork::nn = nullptr;
ifstream* IFile::f = NULL ;
const string Layer::relu = "RELU";
const string Layer::sigmoid = "SIGMOID";
const string Layer::tanh = "TANH";
const string Layer::softmax = "SOFTMAX";
static void corupptedFile() {
	cout << "Corrupted File";
	abort();
}
bool IFile::isOpen() {
	return f->is_open();
}
void IFile::Open(string file) {
	f->open(file);
}
string IFile::getline() {
	string line;
	std::getline(*f, line);
	return line;
}
void IFile::close() {
	f->close();
}
void IFile::setPtr(ifstream* ptr) {
	f = ptr;
}
vector<float> IFile::getinputsVec() {
	std::string line;
	std::getline(*f, line);
	stringstream ss(line);
	vector<float> res;
	string temp_str;

	while (std::getline(ss, temp_str, ',')) { //use comma as delim for cutting string
		res.push_back(std::stof(temp_str));
	}
	return res;
}
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
	_io_type(io_type),
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
	else {
		weight = IFile::getinputsVec();
		if (weight.size() != ninputs)
			corupptedFile();
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
	else {
		biasVec = IFile::getinputsVec();
		if (biasVec.size() != no_precptron) {
			corupptedFile();
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
	else {
		if (!IFile::isOpen())
			abort();
		string line;
		line = IFile::getline();
		_no_layers = std::stoi(line);
		line = IFile::getline();
		_no_inputs = std::stoi(line);
		line = IFile::getline();
		_no_outputs = std::stoi(line);
	}
}

int NeuralNetwork::get_no_precptron(int i) {
	int n = 0;
	if (_io_type == CONSOLE) {
		cout << "No of precptron in layer " << i;
		cin >> n;
	}
	else {
		string line = IFile::getline();
		n = std::stoi(line);
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

		ACT_FUNC func = Layer::getActFunc(i + 1, _io_type);
		Layer* l = new Layer(func, prev, no_precptron, i + 1, _io_type);
		network.push_back(l);
		if (prev)
			prev->setNextLayer(l);
	}
}

std::vector<Layer*>& NeuralNetwork::getLayers() {
	return network;
}

ACT_FUNC Layer::getActFunc(int layer_no, IO_TYPE _io_type) {
	ACT_FUNC func;
	if (_io_type == IO_TYPE::CONSOLE) {
		cout << "Activation functon for layer" << layer_no << "1:Relu 2:Sigmoid 3:TANH 4:SoftMax " << endl;
		cin >> func;
	}
	else {
		string line = IFile::getline();
		if (line.compare(Layer::relu) == 0) {
			func = ACT_FUNC::RELU;
		} else if (line.compare(Layer::sigmoid) == 0) {
			func = ACT_FUNC::SIGMOID;
		}else if (line.compare(Layer::tanh) == 0) {
			func = ACT_FUNC::TANH;
		}else if (line.compare(Layer::softmax) == 0) {
			func = ACT_FUNC::SOFTMAX;
		}

	}
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
	insts.push_back(ip);
	NhookPin(ip->input_pins[0], i1);
	NhookPin(ip->input_pins[1], i2);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::createMul(Net* np, float val) {
	Net* onp = createNet();
	Net* cnp = createNet(true, val);
	Instance* ip = new Instance(InstType::mult);
	insts.push_back(ip);
	NhookPin(ip->input_pins[0], np);
	NhookPin(ip->input_pins[1], cnp);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::addBias(Net* np, float val) {
	Net* onp = createNet();
	Net* cnp = createNet(true, val);
	Instance* ip = new Instance(InstType::add);
	insts.push_back(ip);
	NhookPin(ip->input_pins[0], np);
	NhookPin(ip->input_pins[1], cnp);
	NhookPin(ip->output_pin, onp);
	return onp;
}

Net* Netlist::createActFunc(Net* np, Net* onp,ACT_FUNC f) {
	if(!onp)
		onp = createNet();
	Instance* ip = new Instance(f);
	insts.push_back(ip);
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
	n = Name::getUniqueName(NType::Instance);
	func = f;
	type = InstType::actFunc;
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
	return NULL;
}

NetListWriter::NetListWriter(Netlist* nl) {
	std::string file;
	cin >> file;
	ofs.open(file);
	this->nl = nl;
	this->writePort();
	this->writeNets();
	this->assignConst();
	this->writeInst();
	ofs << std::endl << "endmodule";
	ofs.close();
}

void NetListWriter::writePort() {
	ofs << "module(";
	std::stringstream ss;
	int i = 0;
	for (auto p : nl->inports) {
		if (i != 0)
			ofs << ",";
		std::string port = Name::getNameStr(p.first);
		ofs << port;
		ss << "input [31:0]" << port <<";" << std::endl;
		i++;
	}
	for (auto p : nl->outports) {
		ofs << ",";
		std::string port = Name::getNameStr(p.first);
		ofs << port;
		ss << "output [31:0]" << port <<";" << std::endl;
	}
	ofs << ");" << std::endl;
	ofs << ss.str() << std::endl;

}

void NetListWriter::writeNets() {
	for (auto& n : nl->nets) {
		if (n.second->isPort)
			continue;
		ofs << "wire [31:0]" << Name::getNameStr(n.first) <<" ;" << std::endl;
	}
}

void NetListWriter::writeInst() {
	for (auto i : nl->insts) {
		std::string instName = Name::getNameStr(i->n);
		std::string mod;
		Instance* inst = i;
		switch (inst->type) {
		case InstType::add:
			mod = "fadd";
			break;
		case InstType::mult:
			mod = "fmult";
			break;
		case InstType::reg:
			mod = "freg";
			break;
		case InstType::actFunc:
		{
			ACT_FUNC f = inst->func;
			switch (f) {
			case ACT_FUNC::RELU:
				mod = "relu";
				break;
			case ACT_FUNC::SIGMOID:
				mod = "sigmoid";
				break;
			case ACT_FUNC::SOFTMAX:
				mod = "smax";
				break;
			case ACT_FUNC::TANH:
				mod = "tanh";
				break;
			}
			break;
		}
		}
		ofs << mod << " " << instName << "(";
		for (auto& p : inst->input_pins) {
			std::string pinName = Name::getNameStr(p->n);
			std::string netName = Name::getNameStr(p->np->n);
			ofs << "." << pinName << "(" << netName << "),";
		}
		std::string opinName = Name::getNameStr(inst->output_pin->n);
		std::string opinNetName = Name::getNameStr(inst->output_pin->np->n);
		ofs << "." << opinName << "(" << opinNetName << "));" << std::endl;
	}
}
void NetListWriter::assignConst() {
	for (auto c : Net::constMap) {
		float val = c.first;
		std::string netName = Name::getNameStr(c.second->n);
		std::string bVal = NetListWriter::getConstInBinary(val);
		ofs << "assign " << netName << " = " << bVal <<" ;" << std::endl;
	}
}
std::string NetListWriter::getConstInBinary(float val) {
	bool sign = (val < 0) ? true : false;
	val = (sign) ? (-val) : val;
	int int_val = val;
	float decimal_val = val - (float)int_val;
	std::string dec = "";
	for (int i = 0; i < 11; i++) {
		decimal_val = decimal_val * 2;
		if (decimal_val > 1) {
			dec = dec + "1";
			decimal_val = decimal_val - 1;
		}
		else {
			dec = dec + "0";
		}
	}
	while (dec.length() < 20) {
		dec = "0" + dec;
	}
	std::string integer = "";
	while (int_val > 0) {
		if (int_val % 2 == 1) {
			integer = "1" + integer;
		}
		int_val = int_val / 2;
	}
	if (integer.length() > 20) {
		cout << "weight/bias doesn't fits in 32 bit fixed point number";
		abort();
	}
	stringstream ss;
	ss << "32'b" << (sign) ? "1" : "0";
	ss << integer;
	ss<< dec;
	return ss.str();
}
int main( int argc,char* argv[])
{
	NeuralNetwork* network = NULL;
	ifstream fs;
	if (argc > 3) {
		cout << "Unknown argument";
		abort();
	} else if (argc == 3) {
		string arg1 = argv[1];
		if (arg1.compare("--file") == 0 || arg1.compare("-f") == 0) {
			string file = argv[2];
			fs.open(file);
			IFile::setPtr(&fs);
			network = NeuralNetwork::getInstance(IO_TYPE::File);
		}
		else {
			cout << "Unknown argument";
			abort();
		}
	}
	else if (argc == 2) {
		string arg1 = argv[1];
		if (arg1.compare("--help") == 0 || arg1.compare("-h") == 0) {
			cout << "--file" << "Data File";
			abort();
		}
		else {
			cout << "Unknown argument";
			abort();
		}
	}
	else {
		network = NeuralNetwork::getInstance(IO_TYPE::CONSOLE);
	}
	network->createNetwork();
	Netlist* nl = new Netlist(network);
	NetListWriter nw(nl);
	return 0;
}
