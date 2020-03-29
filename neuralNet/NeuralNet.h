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
        void feedforward(const vector<float> &input);
        void backprop(const vector<float> &result);
        vector<float> getOutput(void);
        vector<vector<vector<float>>> getTheta(void);
        void setTheta(vector<vector<vector<float>>> &theta);
};

class Neural {
    private:
        const float learningRate = 0.03;
        const float alpha = 0.1;
        float activation;  /*0-1*/
        float gradient;   /*error between real and estimated value*/
        vector<float> theta;
        vector<float> deltaTheta;
        unsigned myIndex;
        float sigmoid(const float &input);
        float derivativeSigmoid(const float &input);
    public:
        Neural(const unsigned &nextLayer, const unsigned &activeVal, unsigned &index);
        float getTheta(const unsigned &thetaPos);
        const float getActivation(void);
        void setActivation(const float &value);
        float feedForwardCal(const layer &prevLayer);
        void calGradient(const float &result);
        void calHiddenGradient(const layer nextLayer);
        void updateTheta(layer &prevLayer);
        void setThetaManual(const vector<float> &iTheta);
};

#endif