import cv2
import numpy as np
import pygame
from pygame.locals import *
import neural_network
import copy


'''
CONSTANT VARIABLES
'''
GAME_WIDTH = 1024
GAME_HEIGHT = 786
WORLD_BORDER = 10

class Game(object):
    '''
    Frame is the final image which is shown where everything is drawn on

    Attributes:
    width: width of the frame
    height: height of the frame
    ball: a ball object from the Ball class
    '''
    ball = None
    player1 = None
    player2 = None
    windowName = 'Pong'
    FRAMES = 480
    exit = False


    computerAI = neural_network.Neural_network()

    def __init__(self, width, height, showGame=True, letAIPlay=False):
        #define if frames are drawn or game runs in backround
        self.showGame = showGame
        self.letAIPlay = letAIPlay

        if(self.showGame):
            #initialize pygame
            pygame.init()

            #set window size
            self.gameDisplay = pygame.display.set_mode((width,height))

        #set field to black backround
        self.background_color = (0,0,0)

        #set the clock
        self.clock = pygame.time.Clock()

    def update(self, keyInputs, aiInput=0):

        #check for EXIT
        if keyInputs != [] and keyInputs[K_ESCAPE]:
            self.exitGame()

        #check for Restart
        if keyInputs != [] and keyInputs[K_r]:
            self.newRound(1)

        #update Ball
        self.ball.update(self.player1, self.player2)

        #update players
        self.player1.update(keyInputs,aiInput)
        self.player2.update(keyInputs,aiInput)

        #update score
        gameOver = self.checkForPoint()

        #update frame
        if(self.showGame):
            self.draw()

        if(gameOver):
            return True
        return False

    def exitGame(self):
        self.exit = True
        print("Exiting the game")
        pygame.display.quit()
        pygame.quit()
        exit(0)

    def getCurrentState(self):
        ##TODO: save in format so its int and floats not this shit
        saveArr = self.normalizeCordinates()
        return saveArr

    def draw(self):
        #draw backgound
        self.gameDisplay.fill(self.background_color)

        #draw Ball
        self.drawBall()

        #draw players
        self.drawPlayer(self.player1)
        self.drawPlayer(self.player2)

        #draw score

        #draw the rest

    def drawBall(self):
        #to int for safety
        x = int(self.ball.pos_x)
        y = int(self.ball.pos_y)
        r = int(self.ball.radius)
        c = self.ball.color

        #draw ball on frame with above given measurments
        pygame.draw.circle(self.gameDisplay, c, (x, y), r)

    def drawPlayer(self, player):
        #to int for safety
        x1 = int(player.pos_x)
        y1 = int(player.pos_y)
        x2 = int(player.width)
        y2 = int(player.height)

        #draw rect
        pygame.draw.rect(self.gameDisplay, player.color, ((x1,y1),(x2,y2)))

    def getCurrentBounceCount(self):
        return self.ball.bounceCount

    def checkForPoint(self) -> bool:
        x_new, _ = self.ball.calcFuturePos()
        #check for player 1
        if(x_new < WORLD_BORDER):
            self.addPoint(self.player1)
            self.newRound(1)
            return True

        if(x_new > GAME_WIDTH - WORLD_BORDER):
            self.addPoint(self.player2)
            self.newRound(2)
            return True

        return False

    #TODO: auslagern!
    def computerThink(self):
        data = self.normalizeCordinates()
        data = data.reshape(1,6,1)
        actions = self.computerAI.run(data)
        if(actions[0][0] > actions[0][1]):
            if(self.player2.speed > 0):
                self.player2.speed = -self.player2.speed
            print("MOVE UP")
        else:
            if(self.player2.speed < 0):
                self.player2.speed = -self.player2.speed
            print("MOVE DOWN")

    def newRound(self, winner=1):
        self.ball.set_init()
        if(winner != 1):
            self.ball.bounceX()

        self.player1.set_init_cords()
        self.player2.set_init_cords()
        self.ball.bounceCount = 0

    def addPoint(self, player):
        player.score += 1

    #TODO: auslagern
    def normalizeCordinates(self):
        # normalized array consists of:
        # ball_x cord, ball_y cord, ball_x speed, ball_y speed, player_x cord, player_y cord
        # since there is no max ball speed it will just be divided by 10
        # normalized from 0 to 1
        normArr = np.zeros((1,6))
        normArr[0][0] = self.ball.pos_x/GAME_WIDTH
        normArr[0][1] = self.ball.pos_y/GAME_HEIGHT
        normArr[0][2] = self.ball.speed[0]/10
        normArr[0][3] = self.ball.speed[1]/10
        normArr[0][4] = self.player1.pos_y/GAME_WIDTH
        normArr[0][5] = self.player2.pos_y/GAME_HEIGHT
        return normArr

    #the everlasting loop while running the game
    def run_loop(self):

        print("START RUN LOOP")
        while not self.exit:
            self.run_frame()

        print("EXIT RUN LOOP")

    def run_frame(self, aiInput=0):
        if(self.showGame):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exitGame()
            #check keyboard input
            pressedKeys = pygame.key.get_pressed()
        else:
            pressedKeys = []

        gameOver = self.update(pressedKeys, aiInput)

        if(self.showGame):
            pygame.display.update()
            self.clock.tick(self.FRAMES)

        return gameOver

class Ball(object):
    '''
    The ball for playing

    Attributes:
        pos_x: the x pos of the ball
        pos_y: the y position of the ball
        radius: radius of the ball in px
        color: color is in RGB
        start_direction: the direction the ball starts (from direction enum)
    '''
    #[x,y] speed
    init_speed = [2.0, 0.5]


    def __init__(self, radius, color):
        self.radius = radius
        self.color = color
        self.bounceCount = 0

        #only for x currently
        self.set_init()

    def calcFuturePos(self):
        x_new = self.pos_x + self.speed[0]
        y_new = self.pos_y + self.speed[1]
        return [x_new, y_new]

    def update(self, player1, player2):
        #check velocity

        #calc new direction

        #calc future position
        x_new, y_new = self.calcFuturePos()

        #check boundaries and calc new speed
        if(self.checkPlayer1Contact(player1, x_new, y_new)):
            self.bouncePlayer(player1, y_new)

        if(self.checkPlayer2Contact(player2, x_new, y_new)):
            self.bouncePlayer(player2, y_new)

        self.checkGameBoundaries(x_new, y_new)

        #calc true new direction
        x_new, y_new = self.calcFuturePos()

        #set new pos
        self.pos_x = x_new
        self.pos_y = y_new

    def bounceX(self):
        self.speed[0] = -self.speed[0]

    def bounceY(self):
        self.speed[1] = -self.speed[1]

    def bouncePlayer(self, player, y_new):
        #increases bounce count
        self.bounceCount += 1
        magnitude = player.calcMagnitude(y_new)
        self.speed[1] += magnitude[1]
        if(self.speed[0] > 0):
            self.speed[0] = -self.speed[0] * magnitude[0]
        else:
            self.speed[0] = -self.speed[0] * magnitude[0]

    #TODO: !NOT WITH GAME_WIDTH, SOMETHING DYNAMIC
    def checkGameBoundaries(self, x_new, y_new):
        # X direction
        if(x_new < 0 or x_new > GAME_WIDTH):
            self.bounceX()

        # Y direction
        if(y_new < 0 or y_new > GAME_HEIGHT):
            self.bounceY()

    def checkPlayer1Contact(self, player, x_new, y_new):
        #check contact to player1
        if(x_new <= player.pos_x + player.width and y_new > player.pos_y \
           and y_new < player.pos_y + player.height):
            return True
        return False

    def checkPlayer2Contact(self, player, x_new, y_new):
        #check contact to player2
        if(x_new >= player.pos_x and y_new > player.pos_y \
           and y_new < player.pos_y + player.height):
            return True
        return False

    def set_init(self):
        self.pos_x = int(GAME_WIDTH/2)
        self.pos_y = int(GAME_HEIGHT/2)

        self.speed = copy.deepcopy(self.init_speed)


class Player(object):
    """
    The class for a player Object

    Attributes:
    pos_x: the x pos of the rect
    pos_y: the y position of the rect
    height: height of the rect
    width: the width of the rect
    color: color in RGB
    type: if its 1=player1 or 2=player2

    """
    # TODO: play around with player speed, is quite fun
    init_speed = 5
    score = 0

    def __init__(self, height, width, color, type_int):
        self.height = height
        self.width = width
        self.color = color
        self.type_int = type_int
        self.speed = 0

        self.set_init_cords()

    def update(self, keyInputs, aiInput):
        #get curent y
        y = self.pos_y

        #calc new speed
        self.updateSpeed(keyInputs, aiInput)

        #calc new pos
        y_hat = y + self.speed
        #check boundaries HARD CODED GAME_HEIGHT, not so nice
        if(y_hat > 0 and y_hat + self.height < GAME_HEIGHT):
            #set new pos
            self.pos_y = y_hat
        # else:
        #	 print("Trying to move out of bounds, pos x:{} y:{}"\
        #              .format(self.pos_x, self.pos_y))
        return

    def updateSpeed(self, keyInputs, aiInput):
        # Enable user to overrite AI inputs
        if(not np.any(keyInputs)):
            if aiInput==0:          # 0 for up
                self.speed = -self.init_speed
            if aiInput==1:          # 1 for stay
                self.speed = 0
            if aiInput==2:          # 2 for top
                self.speed = self.init_speed

        else:
            assert keyInputs

            if keyInputs[K_UP]:
                self.speed = -self.init_speed

            if keyInputs[K_DOWN]:
                self.speed = self.init_speed

            if not keyInputs[K_DOWN] and not keyInputs[K_UP]:
                self.speed = 0

    #TODO: CAREFUL WITH CALC BECAUSE IT SCALES ON MORE PIXELS
    def calcMagnitude(self, y):
        '''
        calculates the magnitude for the position y given
        the magnitude is a 2D array with [x,y] magnitude
        '''

        #diff to middle
        magnitude = abs(self.pos_y - y) - self.height/2
        #normalize = [-1,1]
        magnitude = magnitude/(self.height/2)
        #scale numbers are randomly picked can be changed for tweaking
        #TODO: tweaking parameters
        y_magnitude = magnitude**3 * 10
        x_magnitude = abs(magnitude) + 0.8
        return [x_magnitude, y_magnitude]

    def set_init_cords(self):
        #if player1
        if(self.type_int == 1):
            self.pos_x = WORLD_BORDER
        else: #player2
            self.pos_x = GAME_WIDTH - WORLD_BORDER - self.width

        self.pos_y = int(GAME_HEIGHT/2 - self.height/2)


def initStandardGame(showGame=True,letAIPlay=False):
    game = Game(GAME_WIDTH, GAME_HEIGHT, showGame, letAIPlay)
    #x,y,r,c
    ball = Ball(5,(255,255,255))
    player1 = Player(100,20,(255,255,255),1)
    player2 = Player(100,20,(255,255,255),2)
    game.player1 = player1
    game.player2 = player2
    game.ball = ball

    return game
