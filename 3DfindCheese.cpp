#include "3DfindCheese.h"

using namespace std;

/*****************************************************
	SUB FUNCTION
*****************************************************/
void delay(unsigned delaytime) {
	unsigned cnt = 0;
	do {
		cnt++;
	} while(cnt < delaytime);
}

unsigned maxIndex(vector<float> vectorInput) {
	/*find maximum element*/
	vector<unsigned> max {0};
	unsigned max_index = 0;
	for(unsigned indexQvalue = 1; indexQvalue < vectorInput.size(); indexQvalue++) {
		if(vectorInput[max[0]] < vectorInput[indexQvalue]) {
			max.clear();
			max.push_back(indexQvalue);
		}
		else if(vectorInput[max[0]] == vectorInput[indexQvalue]) {
			max.push_back(indexQvalue);
		}
	}
	max_index = max[rand()%max.size()];
	return max_index;
}

/*****************************************************
	CLASS DEFINATION
*****************************************************/
/***************Players Class**************/
Players::Players(const vector<unsigned> &topology):playerNetwork(topology), trainingNetwork(topology) {
	/*do nothing but initialize neural network*/
}

void Players::updatePlayerPos(const unsigned &X,const unsigned &Y) {
	prevPos = curPos;
	curPos.X = X;
	curPos.Y = Y;
}

void Players::initPlayer(const unsigned &X,const unsigned &Y, const unsigned &w, const unsigned &h) {
	originPos.X = X;
	originPos.Y = Y;
	curPos.X = X;
	curPos.Y = Y;
	/*create Q value table*/
	if(0 == q_value.size()) {
		for(unsigned i = 0; i < w; i++) {
			q_value.push_back(vector<stateAction>());
			for(unsigned j = 0; j < h; j++) {
				stateAction tempState = {0, 0, 0, 0};
				q_value[i].push_back(tempState);
			}
		}
	}
}

pos Players::getCurPos(void) const{
	return curPos;
}

void Players::reset(void) {
	static unsigned cnt = 0;
	curPos.X = originPos.X;
	curPos.Y = originPos.Y;
	if(10 > cnt) {
		cnt++;
	}
	else {
		updateTrainingNet();
		cnt = 0;
	}
}

actions Players::playGame(const unsigned &w, const unsigned &h) {
	srand(time(NULL));
	float prop = (float)rand() / RAND_MAX;
	actions chosenAct = LEFT;
	vector<unsigned> listActionWithSameProp;
	if(prop > epsilon) {
		chosenAct = (actions)(rand()%(unsigned)NUM_ACTION);
	}
	else {
		/*main strategy: choose action with highest probability*/
#ifndef DEEP_LEARNING
		chosenAct = chooseActStrategy(curPos.X, curPos.Y);
#else
		vector<float> input;
		vector<float> inputLeft;
		vector<float> inputRight;
		vector<float> inputUp;
		vector<float> inputDown;
		input.push_back(rewardPosX);
		input.push_back(rewardPosY);
		input.push_back(purnishPosX);
		input.push_back(purnishPosY);
		input.push_back((float)(curPos.X/w));
		input.push_back((float)curPos.Y/h);
		inputLeft = input;
		playerNetwork.feedforward(input);
		chosenAct = (actions)maxIndex(playerNetwork.getOutput());
#endif
	}

	/*update epsilon*/
	static unsigned locStep = 0;
	if(6000 > locStep) {
		epsilon += 0.0001;
		locStep++;
	}

	return chosenAct;
}

actions Players::chooseActStrategy(const unsigned &stateX, const unsigned &stateY) {
	actions chosenAct = LEFT;
	float maxActionProp = q_value[stateX][stateY].factionProp[0];
	vector<unsigned> listActionWithSameProp;

	/*main strategy: choose action with highest probability*/
	/*default action*/
	listActionWithSameProp.push_back(0);
	for(unsigned index = 1; index < (unsigned)NUM_ACTION; index++) {
		if(maxActionProp < q_value[stateX][stateY].factionProp[index]) {
			maxActionProp = q_value[stateX][stateY].factionProp[index];
			listActionWithSameProp.clear();
			/*store chosen action*/
			listActionWithSameProp.push_back(index);
		}
		else if(maxActionProp == q_value[stateX][stateY].factionProp[index]) {
			listActionWithSameProp.push_back(index);
		}
	}
	chosenAct = (actions)listActionWithSameProp[(rand()%listActionWithSameProp.size())];
	
	return chosenAct;
}

void Players::updateQtable(const float &reward, const actions &prevAction) {
	/*declare current action value in short term*/
	float *curqa_value = &q_value[prevPos.X][prevPos.Y].factionProp[(unsigned)prevAction];
	/*maximum future state*/
	float nextqa_maxValue = q_value[curPos.X][curPos.Y].factionProp[chooseActStrategy(curPos.X, curPos.Y)];

	*curqa_value = *curqa_value + learningRate * (reward + discount * nextqa_maxValue - *curqa_value);
}

void Players::updateReplayMem(const unsigned gameWidth, const unsigned gameHeight, const float reward, actions action) {
	if(200 > replayMem.numOfStoredMem) {
		/*more space to store data*/
		replayMem.numOfStoredMem++;
	}
	if(199 > replayMem.indexUpdatingMem) {
		replayMem.indexUpdatingMem++;
	}
	else {
		replayMem.indexUpdatingMem = 0;
	}
	/*update replay memory*/
	replayMem.mem[replayMem.indexUpdatingMem].prevPosX = (float)prevPos.X / gameWidth;
	replayMem.mem[replayMem.indexUpdatingMem].prevPosY = (float)prevPos.Y / gameHeight;
	replayMem.mem[replayMem.indexUpdatingMem].nextPosX = (float)curPos.X / gameWidth;
	replayMem.mem[replayMem.indexUpdatingMem].nextPosY = (float)curPos.Y / gameHeight;
	replayMem.mem[replayMem.indexUpdatingMem].prevAction = action;
	replayMem.mem[replayMem.indexUpdatingMem].reward = reward;
}

void Players::trainNetwork(void) {
	/*only train when we have enough replay memory*/
	if(replayMem.numOfStoredMem >= 100) {
		randomBatch();
		/*loop all replay memory*/
		for(unsigned senario = 0; senario < 100; senario++){
			vector<float> inputPrevState {
				rewardPosX,
				rewardPosY,
				purnishPosX,
				purnishPosY,
				replayMem.batch[senario].prevPosX,
				replayMem.batch[senario].prevPosY};
			vector<float> inputNextState {
				rewardPosX,
				rewardPosY,
				purnishPosX,
				purnishPosY,
				replayMem.batch[senario].nextPosX,
				replayMem.batch[senario].nextPosY};
			vector<float> vexpectedQvalue;
			float expectedQvalue = 0;
			/*calculate expected value*/
			trainingNetwork.feedforward(inputNextState);
			vexpectedQvalue = trainingNetwork.getOutput();
			/*find maximum element*/
			unsigned max_index = maxIndex(vexpectedQvalue);
			/*calculate Q value*/
			expectedQvalue = replayMem.batch[senario].reward + discount*vexpectedQvalue[max_index];

			/*training network*/
			playerNetwork.feedforward(inputPrevState);
			vexpectedQvalue = playerNetwork.getOutput();
			vexpectedQvalue[replayMem.batch[senario].prevAction] = expectedQvalue;
			playerNetwork.backprop(vexpectedQvalue);
		}
	}
}

void Players::enviUpdate(const pos &reward, const pos &purnish, const unsigned &w, const unsigned &h) {
	rewardPosX = (float)reward.X/w;
	rewardPosY = (float)reward.Y/h;
	purnishPosX = (float)purnish.X/w;
	purnishPosY = (float)purnish.Y/h;
}

void Players::updateTrainingNet(void) {
	vector<vector<vector<float>>> iTheta;
	iTheta = playerNetwork.getTheta();
	trainingNetwork.setTheta(iTheta);
}

void Players::randomBatch(void) {
	NNstate arr1[200];
	NNstate arr2[200];
	unsigned index = 0;
	unsigned numOfMemLeft = replayMem.numOfStoredMem;
	memcpy(arr1, replayMem.mem, numOfMemLeft*sizeof(NNstate_t));
	memcpy(arr2, arr1, numOfMemLeft*sizeof(NNstate_t));
	for(unsigned num = 0; num < 100; num++) {
		index = rand() % numOfMemLeft;
		replayMem.batch[num] = arr1[index];
		numOfMemLeft--;
		if(index != numOfMemLeft) {
			memcpy(&arr1[index], &arr2[index+1], (numOfMemLeft - 1 - index)*sizeof(NNstate_t));
			memcpy(arr2, arr1, numOfMemLeft*sizeof(NNstate_t));
		}
	}
}

/***************Game Class**************/
Game::Game(const unsigned &w,const unsigned &h, const vector<unsigned> &topology):playerOne(topology) {
	width = w;
	height = h;
	/*temporary variable*/
	unsigned playerPosX;
	unsigned playerPosY;
	/*create random seed*/
	srand(time(NULL));
	do {
		negativeReward.X = (rand()%width);
		positiveReward.X = (rand()%width);
		negativeReward.Y = (rand()%height);
		positiveReward.Y = (rand()%height);
		playerPosX = rand() % width;
		playerPosY = rand() % height;
		playerOne.initPlayer(playerPosX, playerPosY, w, h);
		playerOne.enviUpdate(positiveReward, negativeReward, width, height);
	} while(((negativeReward.X == positiveReward.X) \
			&& (negativeReward.Y == positiveReward.Y)) \
		|| ((negativeReward.X == playerPosX) \
			&& (negativeReward.Y == playerPosY)) \
		|| ((positiveReward.X == playerPosX) \
			&& (positiveReward.Y == playerPosY)));
	/*initialize training net*/
	playerOne.updateTrainingNet();
}

void Game::visualGame(void) {
	/*print scoreboard*/
	cout << "Score: " << score << "  |  NumStep: " << numStep << endl;
	for(int i = -1; i <= (int)height; i++) {
		for(int j = -1; j <= (int)width; j++) {
			/* draw upper cover*/
			if((-1 == i) || (height == i)) {
				cout << "#";
			}
			else {
				/* draw left and right cover */
				if((-1 == j) || (width == j)) {
					cout << "#";
				}
				else {
					if((i == negativeReward.Y) && (j == negativeReward.X)) {
						cout << "O";
					}
					else if((i == positiveReward.Y) && (j == positiveReward.X)) {
						cout << "C";
					}
					else if((i == playerOne.getCurPos().Y) && (j == playerOne.getCurPos().X)) {
						cout << "P";
					}
					else {
						cout << "=";
					}
				}
			}
		}
		cout << endl;
	}
}

void Game::gameStsUpdate(void) {
	actions curAct;
	bool endOfEps = false;
	reward = -0.001;
	/*player take action*/
	curAct = playerOne.playGame(width, height);
	numStep++;
	/*update new player position after taking action*/
	if(LEFT == curAct) {
		if(playerOne.getCurPos().X > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X - 1, playerOne.getCurPos().Y);
		}
		else {
			// playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = -1;
		}
	}
	else if(RIGHT == curAct) {
		if(playerOne.getCurPos().X < width - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X + 1, playerOne.getCurPos().Y);
		}
		else {
			// playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = -1;
		}
	}
	else if(UP == curAct) {
		if(playerOne.getCurPos().Y > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y - 1);
		}
		else {
			// playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = -1;
		}
	}
	else {
		if(playerOne.getCurPos().Y < height - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y + 1);
		}
		else {
			// playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = -1;
		}
	}

	if((positiveReward.X == playerOne.getCurPos().X) && (positiveReward.Y == playerOne.getCurPos().Y)) {
		endOfEps = true;
		score++;
		reward = 1;
	}
	else if((negativeReward.X == playerOne.getCurPos().X) && (negativeReward.Y == playerOne.getCurPos().Y)) {
		endOfEps = true;
		score--;
		reward = -1;
	}
#ifndef DEEP_LEARNING
	playerOne.updateQtable(reward, curAct);
#else
	playerOne.updateReplayMem(width, height, reward, curAct);
	playerOne.trainNetwork();
#endif

	if(true == endOfEps) {
		if((-10000 >= score) || (50 <= score)) {
			endOfGame = true;
		}
		else {
			playerOne.reset();
		}
	}
}

bool Game::getGameSts(void) {
	return endOfGame;
}

/*************************************************
 * MAIN
 * **********************************************/
int main(int argc, char** argv) {
	/*choose architect for neural network*/
	vector<unsigned> topology {6, 6, 6, 4};
	if(argc > 2) {
		Game game(stoi(argv[1]), stoi(argv[2]), topology);
		do {
			system("clear");
			game.gameStsUpdate();
			game.visualGame();
			usleep(10000);
		} while(false == game.getGameSts());
	}
	else {
		cout << "missing argument, correct input: %width %height" << endl;
	}
}