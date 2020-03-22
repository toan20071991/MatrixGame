#include <iostream>
#include <stdlib.h>	// random lib
#include <time.h>
#include <unistd.h>
#include <vector>

#ifndef FIND_CHEESE_H
#define FIND_CHEESE_H

using namespace std;

/*****************************************************
	DEFINE
*****************************************************/
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

class Players {
	private:
		pos curPos;
        pos originPos;
        pos prevPos;
		actions action;	/*4 action as: left - right - up - down*/
		float epsilon = 0.9;	/*80% deterministic action*/
        vector<vector<stateAction>> q_value;
        const float learningRate = 0.2;
        const float discount = 0.9;
        actions chooseActStrategy(const unsigned &stateX, const unsigned &stateY);

	public:
		/*update current player position*/
		void updatePlayerPos(const int &X,const int &Y);
        void initPlayer(const int &X, const int &Y, const int &w, const int &h);
		pos getCurPos(void) const;
		actions playGame(void);
        void reset(void);
        void updateQtable(const float &reward, const actions &prevAction);
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
		Game(const int &w,const int &h);
		void visualGame(void);
		void gameStsUpdate(void);
		bool getGameSts(void);
};

#endif