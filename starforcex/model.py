import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger


class AlienCommunicationAnalysisModel(nn.Module):
    """
    A PyTorch implementation of the Alien Communication Analysis Model,
    inspired by Transformers, designed to analyze, detect, and decode
    potential extraterrestrial communications hidden within radio and audio signals.
    """

    def __init__(
        self, signal_length: int, n_fft: int, num_classes: int
    ):
        super(AlienCommunicationAnalysisModel, self).__init__()
        self.signal_length = (
            signal_length  # Length of the input signal
        )
        self.n_fft = n_fft  # Number of FFT points
        self.num_classes = num_classes  # Number of modulation classes

        # Signal Processing Module
        self.signal_processing = SignalProcessingModule(
            n_fft=self.n_fft
        )

        # Modulation Recognition Module
        self.modulation_recognition = ModulationRecognitionModule(
            num_classes=self.num_classes
        )

        # Symbol Decoding Module
        self.symbol_decoder = SymbolDecodingModule()

        # Semantic Understanding Module
        self.semantic_understanding = SemanticUnderstandingModule()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, signal_length)

        Returns:
            Tensor: Output tensor after semantic understanding module
        """
        logger.debug(
            f"Input tensor shape: {x.shape}"
        )  # Shape: (batch_size, signal_length)

        # Signal Processing
        features = self.signal_processing(x)
        logger.debug(
            f"Features tensor shape after signal processing: {features.shape}"
        )

        # Modulation Recognition
        modulation_logits = self.modulation_recognition(features)
        logger.debug(
            f"Modulation logits tensor shape: {modulation_logits.shape}"
        )

        # Symbol Decoding
        symbols = self.symbol_decoder(features)
        logger.debug(
            f"Symbols tensor shape after symbol decoding: {symbols.shape}"
        )

        # Semantic Understanding
        output = self.semantic_understanding(symbols)
        logger.debug(
            f"Output tensor shape after semantic understanding: {output.shape}"
        )

        return output


class SignalProcessingModule(nn.Module):
    """
    Signal Processing Module performing STFT and other transformations.
    """

    def __init__(self, n_fft: int):
        super(SignalProcessingModule, self).__init__()
        self.n_fft = n_fft  # Number of FFT points

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass performing STFT and other signal processing.

        Args:
            x (Tensor): Input tensor of shape (batch_size, signal_length)

        Returns:
            Tensor: Processed features tensor
        """
        # Perform STFT
        stft_result = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=None,
            win_length=None,
            window=torch.hann_window(self.n_fft).to(x.device),
            return_complex=True,
        )
        # stft_result shape: (batch_size, n_fft // 2 + 1, time_steps)
        logger.debug(f"STFT result shape: {stft_result.shape}")

        # Compute magnitude and phase
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        logger.debug(f"Magnitude tensor shape: {magnitude.shape}")
        logger.debug(f"Phase tensor shape: {phase.shape}")

        # Concatenate magnitude and phase along a new channel dimension
        features = torch.stack((magnitude, phase), dim=1)
        # features shape: (batch_size, 2, n_fft // 2 + 1, time_steps)
        logger.debug(
            f"Features tensor shape after concatenation: {features.shape}"
        )

        return features


class ModulationRecognitionModule(nn.Module):
    """
    Modulation Recognition Module classifying modulation schemes.
    """

    def __init__(self, num_classes: int):
        super(ModulationRecognitionModule, self).__init__()
        self.num_classes = num_classes
        # Example CNN layers to process frequency-time features
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass classifying modulation schemes.

        Args:
            x (Tensor): Input features tensor of shape (batch_size, 2, freq_bins, time_steps)

        Returns:
            Tensor: Logits tensor of shape (batch_size, num_classes)
        """
        x = self.conv1(
            x
        )  # Shape: (batch_size, 16, freq_bins, time_steps)
        x = self.bn1(x)
        x = self.relu(x)
        logger.debug(f"Tensor shape after conv1: {x.shape}")

        x = self.conv2(
            x
        )  # Shape: (batch_size, 32, freq_bins, time_steps)
        x = self.bn2(x)
        x = self.relu(x)
        logger.debug(f"Tensor shape after conv2: {x.shape}")

        x = self.pool(x)  # Shape: (batch_size, 32, 1, 1)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 32)
        logger.debug(
            f"Tensor shape after pooling and flattening: {x.shape}"
        )

        logits = self.fc(x)  # Shape: (batch_size, num_classes)
        logger.debug(f"Logits tensor shape: {logits.shape}")

        return logits


class SymbolDecodingModule(nn.Module):
    """
    Symbol Decoding Module using Transformers to decode symbols from signals.
    """

    def __init__(
        self, d_model: int = 128, nhead: int = 8, num_layers: int = 4
    ):
        super(SymbolDecodingModule, self).__init__()
        self.d_model = d_model

        # Linear layer to project input features to d_model size
        self.input_projection = nn.Linear(
            258, d_model
        )  # Project input features to d_model

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final linear layer to output symbol probabilities
        self.fc = nn.Linear(
            self.d_model, self.d_model
        )  # Assume output dimension equals d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass decoding symbols.

        Args:
            x (Tensor): Input features tensor of shape (batch_size, 2, freq_bins, time_steps)

        Returns:
            Tensor: Decoded symbols tensor
        """
        # Reshape x to (batch_size, time_steps, features)
        batch_size, channels, freq_bins, time_steps = x.shape
        x = x.permute(0, 3, 2, 1).reshape(batch_size, time_steps, -1)
        logger.debug(
            f"Tensor shape after reshaping for transformer input: {x.shape}"
        )

        # Project to d_model size
        x = self.input_projection(x)
        logger.debug(
            f"Tensor shape after input projection: {x.shape}"
        )

        # Apply positional encoding
        x = self.positional_encoding(x)
        logger.debug(
            f"Tensor shape after positional encoding: {x.shape}"
        )

        # Transformer Encoder
        x = x.permute(
            1, 0, 2
        )  # Transformer expects input of shape (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        logger.debug(
            f"Tensor shape after transformer encoder: {x.shape}"
        )

        # Output layer
        x = x.permute(
            1, 0, 2
        )  # Back to (batch_size, seq_len, d_model)
        symbols = self.fc(x)
        logger.debug(f"Symbols tensor shape: {symbols.shape}")

        return symbols


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformers.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Tensor with positional encoding added
        """
        x = x * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float)
        )
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class SemanticUnderstandingModule(nn.Module):
    """
    Semantic Understanding Module to interpret decoded symbols.
    """

    def __init__(
        self, d_model: int = 128, nhead: int = 8, num_layers: int = 4
    ):
        super(SemanticUnderstandingModule, self).__init__()
        self.d_model = d_model

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=nhead
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Final output layer
        self.fc = nn.Linear(
            self.d_model, self.d_model
        )  # Adjust as per output needs

    def forward(self, symbols: Tensor) -> Tensor:
        """
        Forward pass for semantic understanding.

        Args:
            symbols (Tensor): Input symbols tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor representing semantic understanding
        """
        # For transformer decoder, we need a memory tensor from encoder
        # Since we don't have an encoder here, we can simulate or use symbols as both tgt and memory

        tgt = symbols.permute(
            1, 0, 2
        )  # (seq_len, batch_size, d_model)
        memory = symbols.permute(1, 0, 2)

        # Apply transformer decoder
        output = self.transformer_decoder(tgt, memory)
        logger.debug(
            f"Tensor shape after transformer decoder: {output.shape}"
        )

        # Output layer
        output = output.permute(
            1, 0, 2
        )  # (batch_size, seq_len, d_model)
        output = self.fc(output)
        logger.debug(f"Final output tensor shape: {output.shape}")

        return output


# # Example of a forward pass
# if __name__ == "__main__":
#     # Set up logging
#     logger.add("debug.log", level="DEBUG")

#     # Define parameters
#     batch_size = 8
#     signal_length = 1024
#     n_fft = 256
#     num_classes = 10  # For modulation recognition
#     d_model = 128

#     # Create random input tensor
#     x = torch.randn(batch_size, signal_length)

#     # Instantiate the model
#     model = AlienCommunicationAnalysisModel(
#         signal_length=signal_length,
#         n_fft=n_fft,
#         num_classes=num_classes,
#     )

#     # Forward pass
#     output = model(x)

#     print("Output shape:", output.shape)
