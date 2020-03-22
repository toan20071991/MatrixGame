#include "3DfindCheese.h"

using namespace std;

/*****************************************************
	SUB FUNCTION
*****************************************************/
void delay(unsigned int delaytime) {
	unsigned int cnt = 0;
	do {
		cnt++;
	} while(cnt < delaytime);
}

/*****************************************************
	CLASS DEFINATION
*****************************************************/
/***************Players Class**************/

void Players::updatePlayerPos(const int &X,const int &Y) {
	prevPos = curPos;
	curPos.X = X;
	curPos.Y = Y;
}

void Players::initPlayer(const int &X,const int &Y, const int &w, const int &h) {
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
	curPos.X = originPos.X;
	curPos.Y = originPos.Y;
}

actions Players::playGame(void) {
	srand(time(NULL));
	float prop = (float)rand() / RAND_MAX;
	actions chosenAct = LEFT;
	vector<unsigned> listActionWithSameProp;
	if(prop > epsilon) {
		chosenAct = (actions)(rand()%(unsigned)NUM_ACTION);
	}
	else {
		/*main strategy: choose action with highest probability*/
		chosenAct = chooseActStrategy(curPos.X, curPos.Y);
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

/***************Game Class**************/
Game::Game(const int &w,const int &h) {
	width = w;
	height = h;
	/*temporary variable*/
	unsigned int playerPosX;
	unsigned int playerPosY;
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
	} while(((negativeReward.X == positiveReward.X) \
			&& (negativeReward.Y == positiveReward.Y)) \
		|| ((negativeReward.X == playerPosX) \
			&& (negativeReward.Y == playerPosY)) \
		|| ((positiveReward.X == playerPosX) \
			&& (positiveReward.Y == playerPosY)));
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
	reward = -0.1;
	/*player take action*/
	curAct = playerOne.playGame();
	numStep++;
	/*update new player position after taking action*/
	if(LEFT == curAct) {
		if(playerOne.getCurPos().X > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X - 1, playerOne.getCurPos().Y);
		}
		else {
			endOfEps = true;
			score--;
			reward = -10;
		}
	}
	else if(RIGHT == curAct) {
		if(playerOne.getCurPos().X < width - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X + 1, playerOne.getCurPos().Y);
		}
		else {
			endOfEps = true;
			score--;
			reward = -10;
		}
	}
	else if(UP == curAct) {
		if(playerOne.getCurPos().Y > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y - 1);
		}
		else {
			endOfEps = true;
			score--;
			reward = -10;
		}
	}
	else {
		if(playerOne.getCurPos().Y < height - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y + 1);
		}
		else {
			endOfEps = true;
			score--;
			reward = -10;
		}
	}

	if((positiveReward.X == playerOne.getCurPos().X) && (positiveReward.Y == playerOne.getCurPos().Y)) {
		endOfEps = true;
		score++;
		reward = 100;
	}
	else if((negativeReward.X == playerOne.getCurPos().X) && (negativeReward.Y == playerOne.getCurPos().Y)) {
		endOfEps = true;
		score--;
		reward = -10;
	}
	if(-1 == reward) {
		playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y);
	}
	playerOne.updateQtable(reward, curAct);

	if(true == endOfEps) {
		if((-500 >= score) || (50 <= score)) {
			endOfGame = true;
		}
		else {
			playerOne.reset();
			numStep = 0;
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
	if(argc > 2) {
		Game game(stoi(argv[1]), stoi(argv[2]));
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