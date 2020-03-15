#include <iostream>
#include <stdlib.h>	// random lib
#include <time.h>
#include <unistd.h>

using namespace std;

/*****************************************************
	DEFINE
*****************************************************/
typedef struct position {
	unsigned int X;
	unsigned int Y;
} pos;

enum actions {
	LEFT,
	RIGHT,
	UP,
	DOWN
};

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
	MAIN FUNCTION
*****************************************************/
class Players {
	private:
		pos curPos;
		actions action;	/*4 action as: left - right - up - down*/
		float epsilon = 0;	/*80% deterministic action*/

	public:
		/*update current player position*/
		void updatePlayerPos(int X, int Y);
		pos getCurPos(void);
		actions playGame(void);
};

void Players::updatePlayerPos(int X, int Y) {
	curPos.X = X;
	curPos.Y = Y;
}

pos Players::getCurPos(void) {
	return curPos;
}

actions Players::playGame(void) {
	srand(time(NULL));
	float prop = (float)rand() / RAND_MAX;
	actions chosenAct = LEFT;

	if(prop > epsilon) {
		chosenAct = (actions)(rand()%4);
	}
	else {
		/*main strategy*/
	}

	return chosenAct;
}

class Game {
	private:
		unsigned int width;
		unsigned int height;
		Players playerOne;
		pos negativeReward;
		pos positiveReward;
		bool endOfEpisode = false;
		unsigned int score = 0;
		unsigned int numStep = 0;

	public:
		Game(int w, int h);
		void visualGame(void);
		void gameStsUpdate(void);
		bool getGameSts(void);
};

Game::Game(int w, int h) {
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
		// playerPosX = rand() % width;
		// playerPosY = rand() % height;
		playerPosX = width/2;
		playerPosY = height/2;
		playerOne.updatePlayerPos(playerPosX, playerPosY);
	} while(((negativeReward.X == positiveReward.X) \
			&& (negativeReward.Y == positiveReward.Y)) \
		|| ((negativeReward.X == playerPosX) \
			&& (negativeReward.Y == playerPosY)) \
		|| ((positiveReward.X == playerPosX) \
			&& (positiveReward.Y == playerPosY)));
}

void Game::visualGame(void) {
	/*print scoreboard*/
	cout << "NumStep: " << numStep << endl;
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

	/*player take action*/
	curAct = playerOne.playGame();
	numStep++;
	/*update new player position after taking action*/
	if(LEFT == curAct) {
		if(playerOne.getCurPos().X > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X - 1, playerOne.getCurPos().Y);
		}
		else {
			endOfEpisode = true;
		}
	}
	else if(RIGHT == curAct) {
		if(playerOne.getCurPos().X < width - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X + 1, playerOne.getCurPos().Y);
		}
		else {
			endOfEpisode = true;
		}
	}
	else if(UP == curAct) {
		if(playerOne.getCurPos().Y > 0) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y - 1);
		}
		else {
			endOfEpisode = true;
		}
	}
	else {
		if(playerOne.getCurPos().Y < height - 1) {
			playerOne.updatePlayerPos(playerOne.getCurPos().X, playerOne.getCurPos().Y + 1);
		}
		else {
			endOfEpisode = true;
		}
	}
}

bool Game::getGameSts(void) {
	return endOfEpisode;
}



int main(int argc, char** argv) {
	if(argc > 2) {
		Game game(stoi(argv[1]), stoi(argv[2]));
		do {
			system("clear");
			game.gameStsUpdate();
			game.visualGame();
			usleep(100000);
		} while(false == game.getGameSts());
	}
	else {
		cout << "missing argument, correct input: %width %height" << endl;
	}
}