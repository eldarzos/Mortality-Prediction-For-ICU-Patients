"""
Module containing the decoders.
"""
from torch import nn


DECODER_DICT = {
    'Mortality': 'Binary',
    'MortalityFFNN': 'BinaryFFNN',
    'Readmission': 'Binary',
    'LOS': 'Regression',
    'LOS7': 'Binary'
}


# All decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type
    # Handle potential KeyError if model_type is not in dict
    decoder_name = DECODER_DICT.get(model_type)
    if decoder_name is None:
        raise ValueError(f"Unknown or unsupported model_type for decoder: {model_type}")

    # Use getattr for safer lookup than eval
    # return eval(f'Decoder{decoder_name}')
    import sys
    current_module = sys.modules[__name__]
    try:
        return getattr(current_module, f'Decoder{decoder_name}')
    except AttributeError:
            raise ValueError(f"Decoder class 'Decoder{decoder_name}' not found in decoders.py")


class DecoderBinary(nn.Module):
    def __init__(self, hidden_dim):
        """Hidden state decoder for binary classification tasks. Outputs LOGITS."""
        super(DecoderBinary, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # Output raw logits, remove sigmoid
        y = self.fc(h).squeeze(-1) # Use squeeze(-1) to only squeeze the last dim if it's 1
        # If h can be (batch, seq, hidden), fc(h) is (batch, seq, 1)
        # squeeze(-1) makes it (batch, seq)
        # If h is (batch, hidden), fc(h) is (batch, 1), squeeze(-1) makes it (batch,)
        # Ensure output shape is consistent for loss function
        return y


class DecoderRegression(nn.Module):
    def __init__(self, hidden_dim):
        """Hidden state decoder for regression tasks."""
        super(DecoderRegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        y = self.fc(h).squeeze(-1) # Use squeeze(-1)
        return y


class DecoderBinaryFFNN(nn.Module):
    # Note: This FFNN decoder wasn't fully implemented in the original snippet
    # Assuming it should also output logits if used for binary tasks
    def __init__(self, hidden_dim, n_layers):
        """Hidden state decoder for binary classification tasks using FFNN. Outputs LOGITS."""
        super(DecoderBinaryFFNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        assert n_layers > 0

        layers = []
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU()) # Add activation between hidden layers
        for _ in range(1, n_layers): # n_layers includes the first hidden layer
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1)) # Final layer outputs 1 logit

        self.ffnn = nn.Sequential(*layers)


    def forward(self, h):
        # Output raw logits
        y = self.ffnn(h).squeeze(-1) # Use squeeze(-1)
        return y

