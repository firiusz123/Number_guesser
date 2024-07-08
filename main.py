import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
BACKGROUND_COLOR = (0, 0, 0)  # Black background
LINE_COLOR = (255, 255, 255)  # White drawing color
LINE_WIDTH = 5
CIRCLE_RADIUS = 20  # Radius of the circle to be drawn

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Drawing App")

# Variables
drawing = False

# Create a surface for drawing
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BACKGROUND_COLOR)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.circle(canvas, LINE_COLOR, event.pos, CIRCLE_RADIUS)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Clear canvas on 'r' key press
                canvas.fill(BACKGROUND_COLOR)
    
    # Update screen
    screen.blit(canvas, (0, 0))  # Draw the canvas onto the screen
    pygame.display.flip()
    screen.fill(BACKGROUND_COLOR)

    # Access the canvas as a NumPy array
    canvas_array = pygame.surfarray.array3d(canvas)
    print(np.shape(canvas_array))  # Print the array for demonstration

# Quit Pygame
pygame.quit()
sys.exit()
