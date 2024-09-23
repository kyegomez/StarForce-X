import torch
from loguru import logger
from starforcex.model import AlienCommunicationAnalysisModel

# Example of a forward pass
if __name__ == "__main__":
    # Set up logging
    logger.add("debug.log", level="DEBUG")

    # Define parameters
    batch_size = 8
    signal_length = 1024
    n_fft = 256
    num_classes = 10  # For modulation recognition
    d_model = 128

    # Create random input tensor
    x = torch.randn(batch_size, signal_length)

    # Instantiate the model
    model = AlienCommunicationAnalysisModel(
        signal_length=signal_length,
        n_fft=n_fft,
        num_classes=num_classes,
    )

    # Forward pass
    output = model(x)

    print("Output shape:", output.shape)
