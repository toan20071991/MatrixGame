#include "3DfindCheese.h"

using namespace std;

/*****************************************************
	SUB FUNCTION
*****************************************************/

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

void Players::initPlayer(const unsigned &X,const unsigned &Y, const unsigned &w, const unsigned &h, const pos &positive, const pos &negative) {
	originPos.X = X;
	originPos.Y = Y;
	curPos.X = X;
	curPos.Y = Y;
	gameBoardSize.X = w;
	gameBoardSize.Y = h;
	food = positive;
	hole = negative;
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

actions Players::playGame(void) {
	float prop = (float)rand() / RAND_MAX;
	actions chosenAct = LEFT;
	if(prop > epsilon) {
		chosenAct = (actions)(rand()%(unsigned)NUM_ACTION);
	}
	else {
		/*main strategy: choose action with highest probability*/
		chosenAct = chooseActStrategy(curPos.X, curPos.Y);
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
#ifndef DEEP_LEARNING
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
#else
	vector<float> input;
	vector<float> iq_val;
	vector<float> inputLeft {1, 0, 0, 0};
	vector<unsigned> veCurPos = prepareInputForNN(curPos);
	
	/*estimate Left action*/
	input.reserve(4 + veCurPos.size());
	input.insert(input.end(), inputLeft.begin(), inputLeft.end());
	input.insert(input.end(), veCurPos.begin(), veCurPos.end());
	playerNetwork.feedforward(input);
	iq_val.push_back(playerNetwork.getOutput()[0]);

	/*estimate Right action*/
	input[0] = 0;
	input[1] = 1;
	playerNetwork.feedforward(input);
	iq_val.push_back(playerNetwork.getOutput()[0]);

	/*estimate Up action*/
	input[1] = 0;
	input[2] = 1;
	playerNetwork.feedforward(input);
	iq_val.push_back(playerNetwork.getOutput()[0]);

	/*estimate Down action*/
	input[2] = 0;
	input[3] = 1;
	playerNetwork.feedforward(input);
	iq_val.push_back(playerNetwork.getOutput()[0]);

	chosenAct = (actions)maxIndex(iq_val);
#endif
	return chosenAct;
}

void Players::updateQtable(const float &reward, const actions &prevAction) {
	/*declare current action value in short term*/
	float *curqa_value = &q_value[prevPos.X][prevPos.Y].factionProp[(unsigned)prevAction];
	/*maximum future state*/
	float nextqa_maxValue = q_value[curPos.X][curPos.Y].factionProp[chooseActStrategy(curPos.X, curPos.Y)];

	*curqa_value = *curqa_value + learningRate * (reward + discount * nextqa_maxValue - *curqa_value);
}

void Players::updateReplayMem(const float reward, actions action) {
	unsigned ucnt = 0;
	do {
		if(MEMORY_POOL > replayMem.numOfStoredMem) {
			/*more space to store data*/
			replayMem.numOfStoredMem++;
		}
		if(MEMORY_POOL - 1 > replayMem.indexUpdatingMem) {
			replayMem.indexUpdatingMem++;
		}
		else {
			replayMem.indexUpdatingMem = 0;
		}
		/*update replay memory*/
		replayMem.mem[replayMem.indexUpdatingMem].prevPos = prevPos;
		replayMem.mem[replayMem.indexUpdatingMem].nextPos = curPos;
		replayMem.mem[replayMem.indexUpdatingMem].prevAction = action;
		replayMem.mem[replayMem.indexUpdatingMem].reward = reward;

		ucnt++;
	}while((1 == reward) && (20 > ucnt));
}

void Players::trainNetwork(void) {
	/*only train when we have enough replay memory*/
	if(replayMem.numOfStoredMem >= BATCH_SIZE) {
		randomBatch();

		/*loop all replay memory*/
		for(unsigned senario = 0; senario < BATCH_SIZE; senario++){
			vector<float> inputPrevState {0, 0, 0, 0};
			vector<float> inputNextState {0, 0, 0, 0};
			vector<float> vexpectedQvalue;
			float expectedQvalue = 0;
			unsigned max_index = 0;
			vector<unsigned> veNextState = prepareInputForNN(replayMem.batch[senario].nextPos);
			vector<unsigned> vePrevState = prepareInputForNN(replayMem.batch[senario].prevPos);

			inputPrevState.reserve(4 + vePrevState.size());
			for(unsigned h = 0; h < (unsigned)NUM_ACTION; h++) {
				if(h == (unsigned)replayMem.batch[senario].prevAction) {
					inputPrevState[h] = 1;
				}
			}

			inputPrevState.insert(inputPrevState.end(), vePrevState.begin(), vePrevState.end());

			/*not in terminal state*/
			inputNextState.reserve(4 + veNextState.size());
			inputNextState.insert(inputNextState.end(), veNextState.begin(), veNextState.end());
			if(STEP_POINT == replayMem.batch[senario].reward) {
				/*Left action*/
				inputNextState[0] = 1;
				trainingNetwork.feedforward(inputNextState);
				vexpectedQvalue.push_back(trainingNetwork.getOutput()[0]);

				inputNextState[0] = 0;
				inputNextState[1] = 1;
				trainingNetwork.feedforward(inputNextState);
				vexpectedQvalue.push_back(trainingNetwork.getOutput()[0]);

				inputNextState[1] = 0;
				inputNextState[2] = 1;
				trainingNetwork.feedforward(inputNextState);
				vexpectedQvalue.push_back(trainingNetwork.getOutput()[0]);

				inputNextState[2] = 0;
				inputNextState[3] = 1;
				trainingNetwork.feedforward(inputNextState);
				vexpectedQvalue.push_back(trainingNetwork.getOutput()[0]);
			}
			else {
				/*expected value at terminal state is 0*/
				vexpectedQvalue.push_back(0);
			}

			/*training network*/
			playerNetwork.feedforward(inputPrevState);

			/*find maximum element*/
			max_index = maxIndex(vexpectedQvalue);
			/*calculate Q value*/
			expectedQvalue = replayMem.batch[senario].reward + discount*vexpectedQvalue[max_index];

			vexpectedQvalue.clear();
			vexpectedQvalue.push_back(expectedQvalue);
			playerNetwork.backprop(vexpectedQvalue);
		}
	}
}

void Players::updateTrainingNet(void) {
	vector<vector<vector<float>>> iTheta;
	iTheta = playerNetwork.getTheta();
	trainingNetwork.setTheta(iTheta);
}

void Players::randomBatch(void) {
	NNstate arr1[MEMORY_POOL];
	NNstate arr2[MEMORY_POOL];
	unsigned index = 0;
	unsigned numOfMemLeft = replayMem.numOfStoredMem;
	memcpy(arr1, replayMem.mem, numOfMemLeft*sizeof(NNstate_t));
	memcpy(arr2, arr1, numOfMemLeft*sizeof(NNstate_t));
	for(unsigned num = 0; num < BATCH_SIZE; num++) {
		index = rand() % numOfMemLeft;
		replayMem.batch[num] = arr1[index];
		numOfMemLeft--;
		/*not last element*/
		if(index != numOfMemLeft) {
			memcpy(&arr1[index], &arr2[index+1], (numOfMemLeft - 1 - index)*sizeof(NNstate_t));
			memcpy(arr2, arr1, numOfMemLeft*sizeof(NNstate_t));
		}
	}
}

vector<unsigned> Players::prepareInputForNN(pos target) {
	vector<unsigned> result;
	for(unsigned i = 0; i < gameBoardSize.X * gameBoardSize.Y; i++) {
		if((i == (target.X * gameBoardSize.X + target.Y))
			|| (i == (food.X * gameBoardSize.X + food.Y))
			|| (i == (hole.X * gameBoardSize.X + hole.Y)))
		{
			result.push_back(1);
		}
		result.push_back(0);
	}
	return result;
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
		playerOne.initPlayer(playerPosX, playerPosY, w, h, positiveReward, negativeReward);
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
	cout << "Score: " << score << "  |  Win: " << winGame << "  |  Lose: " << loseGame << "  |  NumStep: " << numStep << endl;
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
	reward = STEP_POINT;
	/*player take action*/
	curAct = playerOne.playGame();
	numStep++;
	/*update new player position after taking action*/
	if(LEFT == curAct) {
		if(playerOne.getCurPos().X > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X - 1, playerOne.getCurPos().Y);
		}
		else {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = DEATH_POINT;
		}
	}
	else if(RIGHT == curAct) {
		if(playerOne.getCurPos().X < width - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X + 1, playerOne.getCurPos().Y);
		}
		else {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = DEATH_POINT;
		}
	}
	else if(UP == curAct) {
		if(playerOne.getCurPos().Y > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y - 1);
		}
		else {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = DEATH_POINT;
		}
	}
	else {
		if(playerOne.getCurPos().Y < height - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y + 1);
		}
		else {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
			endOfEps = true;
			score--;
			reward = DEATH_POINT;
		}
	}

	if((positiveReward.X == playerOne.getCurPos().X) && (positiveReward.Y == playerOne.getCurPos().Y)) {
		endOfEps = true;
		score++;
		reward = REWARD_POINT;
	}
	else if((negativeReward.X == playerOne.getCurPos().X) && (negativeReward.Y == playerOne.getCurPos().Y)) {
		endOfEps = true;
		score--;
		reward = DEATH_POINT;
	}
#ifndef DEEP_LEARNING
	playerOne.updateQtable(reward, curAct);
#else
	playerOne.updateReplayMem(reward, curAct);
	playerOne.trainNetwork();
#endif

	if(true == endOfEps) {
		if((-10000 >= score) || (50 <= score)) {
			endOfGame = true;
		}
		else {
			playerOne.reset();
		}
		if(REWARD_POINT == reward) {
			winGame++;
		}
		else {
			loseGame++;
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
	if(argc > 2) {
		unsigned numNN = (unsigned)stoi(argv[1]) * (unsigned)stoi(argv[2]);
		vector<unsigned> topology {numNN, 8, 4, 1};
		Game game(stoi(argv[1]), stoi(argv[2]), topology);
		do {
			game.gameStsUpdate();
			system("clear");
			game.visualGame();
			// usleep(100);
		} while(false == game.getGameSts());
	}
	else {
		cout << "missing argument, correct input: %width %height" << endl;
	}
}