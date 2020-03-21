/*user define*/
#include "NeuralNet.h"

using namespace std;

#define NUM_TEST_CASE 2000

/*****************************************************
 * main func
 * **************************************************/
double input1[NUM_TEST_CASE];
double input2[NUM_TEST_CASE];
double result[NUM_TEST_CASE];
void createTest(void);
int main(int argc, char** argv) {
    vector<unsigned> topology;
    topology.push_back(stoi(argv[1]));
    topology.push_back(stoi(argv[2]));
    topology.push_back(stoi(argv[3]));
    Net myNet(topology);
    createTest();
    for(unsigned i = 0; i < NUM_TEST_CASE; i++) {
        vector<double> input;
        vector<double> veresult;
        vector<double> output;
        input.push_back(input1[i]);
        input.push_back(input2[i]);
        veresult.push_back(result[i]);
        myNet.feedforward(input);
        output = myNet.getOutput();
        for(unsigned j = 0; j < output.size(); j++) {
            cout << "input: " << input1[i] << ", " << input2[i] << ", ";
            cout << "output: " << output[j] << endl;
        }
        myNet.backprop(veresult);
    }

    return 0;
}

void createTest(void) {
    for(unsigned i = 0; i < NUM_TEST_CASE; i++) {
        if(0 == (i % 4)) {
            input1[i] = 0;
            input2[i] = 0;
            result[i] = 0;
        }
        else if(1 == (i % 4)) {
            input1[i] = 1;
            input2[i] = 0;
            result[i] = 1;
        }
        else if(2 == (i % 4)) {
            input1[i] = 0;
            input2[i] = 1;
            result[i] = 1;
        }
        else {
            input1[i] = 1;
            input2[i] = 1;
            result[i] = 1;
        }
    }
}