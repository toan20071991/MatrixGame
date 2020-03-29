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

void Net::feedforward(const vector<float> &input) {
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

void Net::backprop(const vector<float> &result) {
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

vector<float> Net::getOutput(void) {
    vector<float> result;

    for(unsigned i = 0; i < network.back().size() - 1; i++) {
        result.push_back(network.back().at(i).getActivation());
    }
    return result;
}

vector<vector<vector<float>>> Net::getTheta(void) {
    vector<vector<vector<float>>> result;

    /*loop all layers but output layer*/
    for(unsigned ilayers = 0; ilayers < network.size()-1; ilayers++) {
        result.push_back(vector<vector<float>>());
        /*loop all neural including bias*/
        for(unsigned iNeural = 0; iNeural < network[ilayers].size(); iNeural++) {
            result[ilayers].push_back(vector<float>());
            /*loop through all theta for each neural*/
            for(unsigned iTheta = 0; iTheta < network[ilayers+1].size()-1; iTheta++) {
                result[ilayers][iNeural].push_back(network[ilayers][iNeural].getTheta(iTheta));
            }
        }
    }
    return result;
}

void Net::setTheta(vector<vector<vector<float>>> &theta) {
    /*loop all layers but output layer*/
    for(unsigned ilayers = 0; ilayers < network.size()-1; ilayers++) {
        /*loop all neural including bias*/
        for(unsigned iNeural = 0; iNeural < network[ilayers].size(); iNeural++) {
            network[ilayers][iNeural].setThetaManual(theta[ilayers][iNeural]);
        }
    }
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
        theta.push_back((float)rand()/RAND_MAX);
        deltaTheta.push_back(0);
    }
}

float Neural::getTheta(const unsigned &thetaPos) {
    if(thetaPos >= theta.size()) {
        return theta.back();
    }
    else {
        return theta[thetaPos];
    }
}

const float Neural::getActivation(void) {
    return activation;
}

void Neural::setActivation(const float &value) {
    activation = value;
}

float Neural::feedForwardCal(const layer &prevLayer) {
    float result = 0;
    /*loop through previous layer*/
    for(unsigned i = 0; i < prevLayer.size(); i++) {
        result += prevLayer[i].activation * prevLayer[i].theta[myIndex];
    }
    /*calculate activation value by sigmoid function*/
    result = sigmoid(result);
    return result;
}

float Neural::sigmoid(const float &input) {
    // return (input/(1 + abs(input)));
    return tanh(input);
}

float Neural::derivativeSigmoid(const float &input) {
    // return (1/((1+abs(input)) * (1 + abs(input))));
    return (1-(input*input));
}

void Neural::calGradient(const float &result) {
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
        float oldDeltaTheta = prevLayer[i].deltaTheta[myIndex];
        float newDeltaTheta = learningRate * prevLayer[i].getActivation() * gradient \
                                + alpha * oldDeltaTheta;
        prevLayer[i].deltaTheta[myIndex] = newDeltaTheta;
        prevLayer[i].theta[myIndex] += newDeltaTheta;
    }
}

void Neural::setThetaManual(const vector<float> &iTheta) {
    theta = iTheta;
}