import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
LINE_WIDTH = 5

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Drawing App")

# Variables
drawing = False
last_pos = (0, 0)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                pygame.draw.line(screen, LINE_COLOR, last_pos, current_pos, LINE_WIDTH)
                last_pos = current_pos

    pygame.display.flip()
    screen.fill(BACKGROUND_COLOR)

# Quit Pygame
pygame.quit()
sys.exit()
