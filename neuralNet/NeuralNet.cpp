#include <stdio.h>
#include <vector>
#include <iostream>

using namespace std;

/*defined*/
class Net;
class Neural;
typedef vector<Neural> layer;

class Net {
    private:
        vector<layer> network;
    public:
        Net(const vector<unsigned> &topology);
        void feedforward(const vector<double> &input);
        void backprop(const vector<double> &result);
        vector<double> getOutput(void);
};

class Neural {
    private:
        double activation;  /*0-1*/
        vector<double> theta;
        double sigmoid(const double &input);
    public:
        Neural(const unsigned &nextLayer, const unsigned &activeVal);
        double getTheta(const unsigned &thetaPos);
        const double getActivation(void);
        void setActivation(const double &value);
        double feedForwardCal(const layer &prevLayer, const unsigned &thetaPos);
};

/**************************************************
 * define class Net
 * ***********************************************/
Net::Net(const vector<unsigned> &topology) {
    srand(time(NULL));
    /*loop for each layers*/
    for(unsigned i = 0; i < topology.size(); i++) {
        /*create neurals for each layers, including bias neural*/
        network.push_back(layer());
        for(unsigned j = 0; j <= topology[i]; j++) {
            if(i == topology.size() - 1) {
                /*output layer doesn't have next layer*/
                network.back().push_back(Neural(0, 0));
            }
            else {
                network.back().push_back(Neural(topology[i+1], j/topology[i+1]));
            }
        }
    }
}

void Net::feedforward(const vector<double> &input) {
    /*loop through each layer*/
    for(unsigned i = 0; i < network.size(); i++) {
        /*loop through each neural inside every layer*/
        for(unsigned j = 0; j < network[i].size() - 1; j++) {
            /*input layer*/
            if(0 == i) {
                network[i][j].setActivation(input[j]);
            }
            /*others layers*/
            else {
                network[i][j].setActivation(network[i][j].feedForwardCal(network[i-1], j));
            }
        }
    }
}

vector<double> Net::getOutput(void) {
    vector<double> result;

    for(unsigned i = 0; i < network.back().size() - 1; i++) {
        result.push_back(network.back().at(i).getActivation());
    }
    return result;
}

/*****************************************************
 * define class Neural
 * **************************************************/
Neural::Neural(const unsigned &nextLayer, const unsigned &activeVal) {
    
    activation = activeVal;
    /*initialize random theta*/
    for(unsigned i = 0; i < nextLayer; i++) {
        theta.push_back((double)rand()/RAND_MAX);
    }
}

double Neural::getTheta(const unsigned &thetaPos) {
    if(thetaPos >= theta.size()) {
        return theta.back();
    }
    else {
        return theta[thetaPos];
    }
}

const double Neural::getActivation(void) {
    return activation;
}

void Neural::setActivation(const double &value) {
    activation = value;
}

double Neural::feedForwardCal(const layer &prevLayer, const unsigned &thetaPos) {
    double result = 0;
    /*loop through previous layer*/
    for(unsigned i = 0; i < prevLayer.size(); i++) {
        result += prevLayer[i].activation * prevLayer[i].theta[thetaPos];
    }
    /*calculate activation value by sigmoid function*/
    result = sigmoid(result);
    return result;
}

double Neural::sigmoid(const double &input) {
    return (input/(1 + abs(input)));
}
/*****************************************************
 * main func
 * **************************************************/
int main(int argc, char** argv) {
    vector<unsigned> topology;
    topology.push_back(stoi(argv[1]));
    topology.push_back(stoi(argv[2]));
    topology.push_back(stoi(argv[3]));
    Net myNet(topology);
    vector<double> input;
    input.push_back(1);
    input.push_back(2);
    input.push_back(3);
    myNet.feedforward(input);
    
    vector<double> result;
    result = myNet.getOutput();
    for(unsigned i = 0; i < result.size(); i++) {
        cout << "output: " << result[i] << endl;
    }
    return 0;
}