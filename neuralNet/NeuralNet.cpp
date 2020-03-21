/*user include*/
#include "NeuralNet.h"

using namespace std;

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
                network.back().push_back(Neural(0, 0, j));
            }
            else {
                network.back().push_back(Neural(topology[i+1], j/topology[i], j));
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
                network[i][j].setActivation(network[i][j].feedForwardCal(network[i-1]));
            }
        }
    }
}

void Net::backprop(const vector<double> &result) {
    /*update gradient for output layer*/
    for(unsigned i = 0; i < result.size(); i++) {
        network.back()[i].calGradient(result[i]);
    }
    /*update gradient for hidden layer*/
    for(unsigned i = network.size() - 2; i > 0; i--) {
        for(unsigned j = 0; j < network[i].size() - 1; j++) {
            network[i][j].calHiddenGradient(network[i+1]);
        }
    }
    /*update theta*/
    for(unsigned i = network.size() - 1; i > 0; i--) {
        for(unsigned j = 0; j < network[i].size() - 1; j++) {
            network[i][j].updateTheta(network[i - 1]);
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
Neural::Neural(const unsigned &nextLayer, const unsigned &activeVal, unsigned &index) {
    
    activation = activeVal;
    gradient = 0;
    myIndex = index;
    /*initialize random theta*/
    for(unsigned i = 0; i < nextLayer; i++) {
        theta.push_back((double)rand()/RAND_MAX);
        deltaTheta.push_back(0);
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

double Neural::feedForwardCal(const layer &prevLayer) {
    double result = 0;
    /*loop through previous layer*/
    for(unsigned i = 0; i < prevLayer.size(); i++) {
        result += prevLayer[i].activation * prevLayer[i].theta[myIndex];
    }
    /*calculate activation value by sigmoid function*/
    result = sigmoid(result);
    return result;
}

double Neural::sigmoid(const double &input) {
    // return (input/(1 + abs(input)));
    return tanh(input);
}

double Neural::derivativeSigmoid(const double &input) {
    // return (1/((1+abs(input)) * (1 + abs(input))));
    return (1-(input*input));
}

void Neural::calGradient(const double &result) {
    gradient = (result - activation);
}

void Neural::calHiddenGradient(const layer nextLayer) {
    for(unsigned i = 0; i < nextLayer.size() - 1; i++) {
        gradient += theta[i] * nextLayer[i].gradient;
    }
    gradient *= derivativeSigmoid(activation);
}

void Neural::updateTheta(layer &prevLayer) {
    for(unsigned i = 0; i < prevLayer.size(); i++) {
        double oldDeltaTheta = prevLayer[i].deltaTheta[myIndex];
        double newDeltaTheta = learningRate * prevLayer[i].getActivation() * gradient \
                                + alpha * oldDeltaTheta;
        prevLayer[i].deltaTheta[myIndex] = newDeltaTheta;
        prevLayer[i].theta[myIndex] += newDeltaTheta;
    }
}