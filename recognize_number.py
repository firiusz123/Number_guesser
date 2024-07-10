import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

# Define the CNN model
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

class Predictor():
    def __init__(self):
        self.model = CNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('mnist_cnn.pth'))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, tensor_img):
        # Ensure the input is a PyTorch tensor
        if not isinstance(tensor_img, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
        
        # Normalize the tensor to values between 0 and 1 if not already normalized
        if tensor_img.max() > 1:
            tensor_img = tensor_img.float() / 255.0

        # Add channel dimension if necessary (assumes grayscale image)
        if len(tensor_img.shape) == 2:  # If shape is (H, W)
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(tensor_img.shape) == 3:  # If shape is (C, H, W)
            tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension
        elif len(tensor_img.shape) == 4 and tensor_img.shape[1] == 3:  # If shape is (B, H, W, C)
            tensor_img = tensor_img.mean(dim=1, keepdim=True)  # Convert RGB to grayscale

        # Resize the tensor to (1, 1, 28, 28)
        resized_tensor = F.interpolate(tensor_img, size=(28, 28), mode='bilinear')

        # Move the tensor to the appropriate device
        resized_tensor = resized_tensor.to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(resized_tensor)

        # Get the predicted class
        predicted_class = output.argmax().item()
        return predicted_class

# Example usage
random_array = np.random.rand(600, 600)

# Convert the random array to a PyTorch tensor
tensor_img = torch.tensor(random_array, dtype=torch.float32)

# Instantiate the predictor
predictor = Predictor()

# Predict the class of the image
predicted_class = predictor.predict(tensor_img)
print(f'Predicted class is {predicted_class}')
