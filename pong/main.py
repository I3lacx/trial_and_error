import cv2
import numpy as np
import pygame
from pygame.locals import *

'''
CONSTANT VARIABLES
'''
GAME_WIDTH = 1024
GAME_HEIGHT = 786
WORLD_BORDER = 10

class Pong(object):
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
    FRAMES = 60
    exit = False

    def __init__(self, width, height):
        #initialize pygame
        pygame.init()

        #set window size
        self.gameDisplay = pygame.display.set_mode((width,height))

        #set field to black backround
        self.background_color = (0,0,0)

        #set the clock
        self.clock = pygame.time.Clock()

    def update(self, keyInputs):

        #check for EXIT
        if keyInputs[K_ESCAPE]:
            self.exitGame()

        #check for Restart
        if keyInputs[K_r]:
            self.newRound(1)

        #update Ball
        self.ball.update(self.player1, self.player2)

        #update players
        self.player1.update(keyInputs)
        self.player2.update(keyInputs)

        #update score
        self.checkForPoint()

        #update frame
        self.draw()

    def exitGame(self):
        self.exit = True
        print("Exiting the game")

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

    def checkForPoint(self):
        x_new, _ = self.ball.calcFuturePos()
        #check for player 1
        if(x_new < WORLD_BORDER):
            self.addPoint(self.player1)
            self.newRound(1)

        if(x_new > GAME_WIDTH - WORLD_BORDER):
            self.addPoint(self.player2)
            self.newRound(2)

    def newRound(self, winner):
        #1 for player1 scored a point, 2 for player2 scored a point
        if(winner == 1):
            self.ball.speed = self.ball.init_speed
        else:
            self.ball.speed = self.ball.init_speed
            self.ball.bounceX()

        self.ball.set_init()
        self.player1.set_init_cords()
        self.player2.set_init_cords()

    def addPoint(self, player):
        player.score += 1

    #the everlasting loop while running the game
    def run(self):
        print("START RUN LOOP")

        while not self.exit:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exitGame()
                    break

            #check keyboard input
            pressedKeys = pygame.key.get_pressed()

            self.update(pressedKeys)

            pygame.display.update()
            self.clock.tick(self.FRAMES)


        print("EXIT RUN LOOP")

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
    init_speed = [4.0, 2.0]

    def __init__(self, radius, color):
        self.radius = radius
        self.color = color

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
        magnitude = player.calcMagnitude(y_new)
        self.speed[1] += magnitude[1]
        print("curr speed :x=",self.speed[0])
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

        # : to copy list
        self.speed = self.init_speed[:]

class Player(object):
    '''
    The class for a player Object

    Attributes:
        pos_x: the x pos of the rect
        pos_y: the y position of the rect
        height: heihgt of the rect
        width: the width of the rect
        color: color in RGB
        type: if its 1=player1 or 2=player2

    '''
    init_speed = 5
    score = 0

    def __init__(self, height, width, color, type_int):
        self.height = height
        self.width = width
        self.color = color
        self.type_int = type_int
        self.speed = self.init_speed

        self.set_init_cords()

    def update(self, keyInputs):
        #get curent y
        y = self.pos_y

        #check movement
        if keyInputs[K_UP] and self.speed > 0:
            self.speed = -self.speed

        if keyInputs[K_DOWN] and self.speed < 0:
            self.speed = -self.speed

        #calc new pos
        y_hat = y + self.speed

        #check boundaries HARD CODED GAME_HEIGHT, not so nice
        if(y_hat > 0 and y_hat + self.height < GAME_HEIGHT):
            #set new pos
            self.pos_y = y_hat
        # else:
        #     print("Trying to move out of bounds, pos x:{} y:{}"\
        #             .format(self.pos_x, self.pos_y))
        return

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
        print("magnitues: x=",x_magnitude,", y=",y_magnitude)
        return [x_magnitude, y_magnitude]

    def set_init_cords(self):
        #if player1
        if(self.type_int == 1):
            self.pos_x = WORLD_BORDER
        else: #player2
            self.pos_x = GAME_WIDTH - WORLD_BORDER - self.width

        self.pos_y = int(GAME_HEIGHT/2 - self.height/2)

def main():
    #init
    print("Initialize")

    game = Pong(GAME_WIDTH, GAME_HEIGHT)
    #x,y,r,c
    ball = Ball(5,(255,255,255))
    player1 = Player(100,20,(255,255,255),1)
    player2 = Player(100,20,(255,255,255),2)
    game.player1 = player1
    game.player2 = player2
    game.ball = ball
    game.run()

if __name__ == "__main__":
    main()
