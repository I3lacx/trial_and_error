import cv2
import numpy as np
import pygame
from pygame.locals import *

'''
CONSTANT VARIABLES
'''
GAME_WIDTH = 1024
GAME_HEIGHT = 786


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

        #update Ball
        self.ball.update(self.player1, self.player2)

        #update players
        self.player1.update(keyInputs)
        #update score

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

        #draw score

        #draw the rest


    def drawBall(self):
        #just to make the code shorter
        x = self.ball.pos_x
        y = self.ball.pos_y
        r = self.ball.radius
        c = self.ball.color

        #draw ball on frame with above given measurments
        pygame.draw.circle(self.gameDisplay, c, (x, y), r)

    def drawPlayer(self, player):
        x1 = player.pos_x
        y1 = player.pos_y
        x2 = player.width
        y2 = player.height

        #draw rect
        pygame.draw.rect(self.gameDisplay, player.color, ((x1,y1),(x2,y2)))

    #the everlasting loop while running the game
    def run(self):
        print("START RUN LOOP")

        while not self.exit:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exitGame()
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

    def __init__(self, x_start, y_start, radius, color, start_speed):
        self.pos_x = x_start
        self.pos_y = y_start
        self.radius = radius
        self.color = color

        #only for x currently
        self.speed = start_speed

    def update(self, player1, player2):
        #get current pos_x
        x = self.pos_x
        y = self.pos_y

        #check velocity

        #calc new direction

        #check contact to players
        if(x + self.speed <= player1.pos_x + player1.width \
            and y > player1.pos_y and y < player1.pos_y + player1.height):
            self.speed = -self.speed


        #check game bounderies !CAREFUL NOT WITH GAME_WIDTH, SOMETHING DYNAMIC
        if(x + self.speed < 0 or x + self.speed > GAME_WIDTH):
            self.speed = -self.speed

        #calc new pos
        x += self.speed

        #set new pos
        self.pos_x = x

class Player(object):
    '''
    The class for a player Object

    Attributes:
        pos_x: the x pos of the rect
        pos_y: the y position of the rect
        height: heihgt of the rect
        width: the width of the rect
        color: color in RGB

    '''
    speed = 5

    def __init__(self, start_x, start_y, height, width, color):
        self.pos_x = start_x
        self.pos_y = start_y
        self.height = height
        self.width = width
        self.color = color

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
        else:
            print("Trying to move out of bounds, pos x:{} y:{}"\
                    .format(self.pos_x, self.pos_y))

        return



def main():
    #init
    print("Initialize")

    game = Pong(GAME_WIDTH, GAME_HEIGHT)
    #x,y,r,c,s
    ball = Ball(0,300,5,(255,255,255),5)
    player1 = Player(20,20,100,20,(255,255,255))
    game.player1 = player1
    game.ball = ball
    game.run()

if __name__ == "__main__":
    main()
