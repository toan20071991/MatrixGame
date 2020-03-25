#include <iostream>
#include <stdlib.h>	// random lib
#include <time.h>
#include <unistd.h>
#include <vector>
#include <string.h>
#include "neuralNet/NeuralNet.h"

#ifndef FIND_CHEESE_H
#define FIND_CHEESE_H

using namespace std;

/*****************************************************
	DEFINE
*****************************************************/
/*deep learning or not*/
#define DEEP_LEARNING

typedef struct position {
	unsigned X;
	unsigned Y;
} pos;

enum actions {
	LEFT,
	RIGHT,
	UP,
	DOWN,
    NUM_ACTION
};

typedef struct stateAction_t {
    float factionProp[NUM_ACTION];
} stateAction;

/*struct for training neural network*/
typedef struct NNstate_t {
	actions prevAction;
	float prevPosX;
	float prevPosY;
	float reward;
	float nextPosX;
	float nextPosY;
} NNstate;

typedef struct NNstateHdl_t {
	unsigned numOfStoredMem = 0;
	unsigned indexUpdatingMem = 0;
	NNstate mem[200];
	NNstate batch[100];
} NNstateHdl;

class Players {
	private:
		pos curPos;
        pos originPos;
        pos prevPos;
		float rewardPosX;
		float rewardPosY;
		float purnishPosX;
		float purnishPosY;
		actions action;	/*4 action as: left - right - up - down*/
		float epsilon = 0.3;	/*deterministic action*/
        vector<vector<stateAction>> q_value;
		NNstateHdl replayMem;
		Net playerNetwork;
		Net trainingNetwork;
        const float learningRate = 0.2;
        const float discount = 0.9;
        actions chooseActStrategy(const unsigned &stateX, const unsigned &stateY);
		void randomBatch(void);

	public:
		Players(const vector<unsigned> &topology);
		/*update current player position*/
		void updatePlayerPos(const unsigned &X,const unsigned &Y);
        void initPlayer(const unsigned &X, const unsigned &Y, const unsigned &w, const unsigned &h);
		pos getCurPos(void) const;
		actions playGame(const unsigned &w, const unsigned &h);
        void reset(void);
        void updateQtable(const float &reward, const actions &prevAction);
		void updateReplayMem(const unsigned gameWidth, const unsigned gameHeight, const float reward, actions action);
		void trainNetwork(void);
		void enviUpdate(const pos &reward, const pos &purnish, const unsigned &w, const unsigned &h);
		void updateTrainingNet(void);
};

class Game {
	private:
		unsigned width;
		unsigned height;
		Players playerOne;
		pos negativeReward;
		pos positiveReward;
		bool endOfGame = false;
		int score = 0;
		unsigned numStep = 0;
        float reward = 0;

	public:
		Game(const unsigned &w,const unsigned &h, const vector<unsigned> &topology);
		void visualGame(void);
		void gameStsUpdate(void);
		bool getGameSts(void);
};

#endif