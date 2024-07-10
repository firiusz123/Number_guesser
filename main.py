import pygame
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from recognize_number import CNN, Predictor



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        layer_size = 32 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=layer_size, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=layer_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.33)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x 


model = CNN()
state_dict = torch.load('mnist_cnn.pth')
# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])














# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
BACKGROUND_COLOR = (0, 0, 0)  # Black background
LINE_COLOR = (255, 255, 255)  # White drawing color
LINE_WIDTH = 5
CIRCLE_RADIUS = 20  # Radius of the circle to be drawn
FONT_SIZE = 25
FONT_COLOR = (255, 255, 255)
FONT_POSITION = (WIDTH - 120, 20)  # Position for the text (top right corner)
FONT_POSITION1 = (WIDTH - 500, 20)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Drawing App")

# Create a font object
font = pygame.font.Font(None, FONT_SIZE)

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
    canvas_array = pygame.surfarray.array3d(canvas)
   
    canvas_array = pygame.surfarray.array3d(canvas)
    canvas_array = np.transpose(canvas_array, (1, 0, 2))  # Pygame's array shape is (width, height, channels)
    canvas_image = transform(canvas_array)
    canvas_image = canvas_image.unsqueeze(0)  # Add batch dimension
    #print(canvas_image.shape) 
    with torch.no_grad():
        output = model(canvas_image)
        _, predicted_digit = torch.max(output, 1)
        print(predicted_digit)
    text_surface = font.render(f"The number predicted is {predicted_digit.item()}", True, FONT_COLOR)
    screen.blit(text_surface, FONT_POSITION1)


    
    #print(np.shape(canvas_array))  # Print the array for demonstration

    # Draw text on the screen (outside canvas)
    text_surface = font.render("Press 'r' to clear", True, FONT_COLOR)
    screen.blit(text_surface, FONT_POSITION)

    pygame.display.flip()
    screen.fill(BACKGROUND_COLOR)

# Access the canvas as a NumPy array


# Quit Pygame
pygame.quit()
sys.exit()
