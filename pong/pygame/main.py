import numpy as np
import pygame
from pygame.locals import *

GAME_WIDTH = 1024
GAME_HEIGHT = 786


pygame.init()


gameDisplay = pygame.display.set_mode((GAME_WIDTH,GAME_HEIGHT))
pygame.display.set_caption('Pong')
clock = pygame.time.Clock()


#Game cycle
crashed = False
while not crashed:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    gameDisplay.fill((0,0,0))
    pygame.draw.rect(gameDisplay, (255,255,255), ((20,20),(100,100)))
    pygame.draw.circle(gameDisplay, (255,255,255), (300,200), 10)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
