import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    # Define the neural network architecture
    def __init__(self):
        super().__init__()
        # Fully connected layers
        self.fc1 = torch.nn.Linear(28*28, 64)  # First layer: input from flattened 28x28 image to 64 nodes
        self.fc2 = torch.nn.Linear(64, 64)     # Second layer: 64 nodes to 64 nodes
        self.fc3 = torch.nn.Linear(64, 64)     # Third layer: 64 nodes to 64 nodes
        self.fc4 = torch.nn.Linear(64, 10)     # Output layer: 64 nodes to 10 nodes (one for each digit)

    def forward(self, x):
        # Forward pass through the network
        x = torch.nn.functional.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = torch.nn.functional.relu(self.fc2(x))  # Apply ReLU activation function after second layer
        x = torch.nn.functional.relu(self.fc3(x))  # Apply ReLU activation function after third layer
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # Apply log softmax on the output layer
        return x

def get_data_loader(is_train):
    # Function to load MNIST data
    to_tensor = transforms.Compose([transforms.ToTensor()])  # Transform the data to tensor format
    data_set = MNIST("", is_train, transform=to_tensor, download=True)  # Download and transform the MNIST dataset
    return DataLoader(data_set, batch_size=15, shuffle=True)  # Return a DataLoader for batch processing

def evaluate(test_data, net):
    # Function to evaluate the network's performance
    n_correct = 0
    n_total = 0
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))  # Forward pass on the test data
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:  # Check if the prediction matches the label
                    n_correct += 1
                n_total += 1
    return n_correct / n_total  # Calculate and return the accuracy

def main():
    # Main function for training and evaluating the network
    train_data = get_data_loader(is_train=True)  # Load training data
    test_data = get_data_loader(is_train=False)  # Load test data
    net = Net()  # Initialize the neural network

    # Evaluate initial (untrained) accuracy
    print("initial accuracy:", evaluate(test_data, net))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Set up the optimizer with learning rate
    for epoch in range(2):  # Train for 2 epochs
        for (x, y) in train_data:
            net.zero_grad()  # Reset gradients
            output = net.forward(x.view(-1, 28*28))  # Forward pass
            loss = torch.nn.functional.nll_loss(output, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))  # Print accuracy after each epoch

    # Visualization of predictions
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))  # Predict the digit
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))  # Display the image
        plt.title("prediction: " + str(int(predict)))  # Show the prediction as the title
    plt.show()

if __name__ == "__main__":
    main()


