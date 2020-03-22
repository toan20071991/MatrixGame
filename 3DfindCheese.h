#include <iostream>
#include <stdlib.h>	// random lib
#include <time.h>
#include <unistd.h>
#include <vector>
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
	float prevPosX;
	float prevPosY;
	actions prevAction;
	float reward;
	float nextPosX;
	float nextPosY;
} NNstate;

typedef struct NNstateHdl_t {
	unsigned numOfStoredMem = 0;
	unsigned indexUpdatingMem = 0;
	NNstate mem[100];
} NNstateHdl;

class Players {
	private:
		pos curPos;
        pos originPos;
        pos prevPos;
		actions action;	/*4 action as: left - right - up - down*/
		float epsilon = 0.9;	/*deterministic action*/
        vector<vector<stateAction>> q_value;
		NNstateHdl replayMem;
		Net playerNetwork;
        const float learningRate = 0.2;
        const float discount = 0.9;
        actions chooseActStrategy(const unsigned &stateX, const unsigned &stateY);

	public:
		Players(const vector<unsigned> &topology);
		/*update current player position*/
		void updatePlayerPos(const int &X,const int &Y);
        void initPlayer(const int &X, const int &Y, const int &w, const int &h);
		pos getCurPos(void) const;
		actions playGame(const unsigned &w, const unsigned &h);
        void reset(void);
        void updateQtable(const float &reward, const actions &prevAction);
		void updateReplayMem(const unsigned gameWidth, const unsigned gameHeight, const float reward, actions action);
		void trainNetwork(void);
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
		Game(const int &w,const int &h, const vector<unsigned> &topology);
		void visualGame(void);
		void gameStsUpdate(void);
		bool getGameSts(void);
};

#endif