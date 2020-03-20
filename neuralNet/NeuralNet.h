#include <stdio.h>
#include <vector>
#include <iostream>
#include <math.h>

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

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
        const double learningRate = 0.3;
        const double alpha = 0.5;
        double activation;  /*0-1*/
        double gradient;   /*error between real and estimated value*/
        vector<double> theta;
        vector<double> deltaTheta;
        unsigned myIndex;
        double sigmoid(const double &input);
        double derivativeSigmoid(const double &input);
    public:
        Neural(const unsigned &nextLayer, const unsigned &activeVal, unsigned &index);
        double getTheta(const unsigned &thetaPos);
        const double getActivation(void);
        void setActivation(const double &value);
        double feedForwardCal(const layer &prevLayer, const unsigned &thetaPos);
        void calGradient(const double &result);
        void calHiddenGradient(const layer nextLayer);
        void updateTheta(layer &prevLayer);
};

#endif