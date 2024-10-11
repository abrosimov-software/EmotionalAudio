from src import RandomEmotionDetector
import torch
import mlflow
import mlflow.pytorch
import os

# Set the experiment name
experiment_name = "emotion_detection"
mlflow.set_experiment(experiment_name)

# Create a new run
with mlflow.start_run():
    # Create a model
    model = RandomEmotionDetector()
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Set the loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Set the number of epochs
    epochs = 10
    # Set the batch size
    batch_size = 32

    # Train the model
    for epoch in range(epochs):
        # Train the model
        for i in range(100):
            # Generate random input data
            x = torch.randn(batch_size, 1, 128)
            y = torch.randint(0, 7, (batch_size,))
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x)
            # Calculate the loss
            loss = criterion(outputs, y)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
        # Log the loss
        mlflow.log_metric("loss", loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    # Save the model
    mlflow.pytorch.log_model(model, "model")
    # Save the model parameters
    torch.save(model.state_dict(), "model.pth")
    # Save the model architecture
    with open("model.txt", "w") as f:
        f.write(str(model))
    # Save the model artifacts
    mlflow.log_artifact("model.pth")
    mlflow.log_artifact("model.txt")
    # Save the model parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_classes", model.num_classes)
    # Save the model parameters
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("criterion", criterion)
    # Save the model parameters
    mlflow.log_param("experiment_name", experiment_name)
    mlflow.log_param("model_name", "RandomEmotionDetector")
    mlflow.log_param("model_type", "pytorch")
    mlflow.log_param("model_version", "1.0.0")
    mlflow.log_param("model_description", "Random classifier for testing purposes")
    mlflow.log_param("model_author", "John Doe")
    mlflow.log_param("model_date", "2022-01-01")
    mlflow.log_param("model_license", "MIT")
