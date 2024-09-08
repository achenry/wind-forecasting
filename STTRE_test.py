import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Read the LUT data
lut_data = pd.read_csv('lut/time_series_results_case_LUT_seed_0.csv')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_dim = 5  # Time, FreestreamWindMag, FreestreamWindDir, TurbineWindMag_0, TurbineWindDir_0
output_dim = 1  # Predicted TurbineYawAngle_0
hidden_dim = 64
num_layers = 3
num_heads = 4
dropout = 0.1
num_epochs = 100
batch_size = 32
learning_rate = 0.001
seq_length = 24  # 24 hours of historical data

# Generate sample wind farm data
def generate_wind_farm_data(num_samples, num_turbines):
    data = []
    for _ in range(num_samples):
        timestamp = np.arange(seq_length)
        for turbine in range(num_turbines):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            wind_speed = np.random.normal(8, 2, seq_length)
            wind_direction = np.random.uniform(0, 360, seq_length)
            data.append(np.column_stack((timestamp, np.full(seq_length, turbine), np.full(seq_length, x), np.full(seq_length, y), wind_speed, wind_direction)))
    return np.vstack(data)

# Generate sample data
num_samples = 1000
num_turbines = 10
data = generate_wind_farm_data(num_samples, num_turbines)

# Prepare input features and target
X = lut_data[['Time', 'FreestreamWindMag', 'FreestreamWindDir', 'TurbineWindMag_0', 'TurbineWindDir_0']].values
y = lut_data['TurbineYawAngle_0'].values

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale the features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape data for sequence input
def reshape_sequences(data, seq_length, step=1):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, step):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

X_train_seq = reshape_sequences(X_train_scaled, seq_length)
X_test_seq = reshape_sequences(X_test_scaled, seq_length)
y_train_seq = reshape_sequences(y_train_scaled, seq_length)
y_test_seq = reshape_sequences(y_test_scaled, seq_length)

# Ensure consistent shapes
min_samples = min(X_train_seq.shape[0], y_train_seq.shape[0])
X_train_seq = X_train_seq[:min_samples]
y_train_seq = y_train_seq[:min_samples]

min_samples = min(X_test_seq.shape[0], y_test_seq.shape[0])
X_test_seq = X_test_seq[:min_samples]
y_test_seq = y_test_seq[:min_samples]

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

# STTRE model definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values
        # (we use dim=-1 to normalize the last dimension)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        x = self.embedding(x)
        x = x + self.position_embedding(x).to(self.device)
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class STTRE(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        super(STTRE, self).__init__()
        self.encoder = Encoder(
            input_dim,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        
        self.device = device
        self.fc_out = nn.Linear(embed_size, 1)  # Output dimension is 1 for regression

    def forward(self, src, src_mask):
        enc_src = self.encoder(src, src_mask)
        out = self.fc_out(enc_src)  # Apply fc_out to all time steps
        return out

# Initialize the model
input_dim = X_train_tensor.shape[2]  # number of features
max_length = seq_length
model = STTRE(
    input_dim,
    hidden_dim,
    num_layers,
    num_heads,
    4,
    dropout,
    max_length,
    device
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size].unsqueeze(-1)  # Add an extra dimension
        
        src_mask = None  # Implement masking if needed
        
        outputs = model(batch_X, src_mask)
        
        # Ensure outputs and batch_y have the same shape
        if outputs.shape != batch_y.shape:
            if len(outputs.shape) == 3 and len(batch_y.shape) == 4:
                batch_y = batch_y.squeeze(-1)
            elif len(outputs.shape) == 4 and len(batch_y.shape) == 3:
                outputs = outputs.squeeze(-1)
        
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / (len(X_train_tensor) // batch_size)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    src_mask = None  # Implement masking if needed
    test_outputs = model(X_test_tensor, src_mask)
    
    # Ensure test_outputs and y_test_tensor have the same shape
    y_test_tensor_expanded = y_test_tensor.unsqueeze(-1)
    if test_outputs.shape != y_test_tensor_expanded.shape:
        if len(test_outputs.shape) == 3 and len(y_test_tensor_expanded.shape) == 4:
            y_test_tensor_expanded = y_test_tensor_expanded.squeeze(-1)
        elif len(test_outputs.shape) == 4 and len(y_test_tensor_expanded.shape) == 3:
            test_outputs = test_outputs.squeeze(-1)
    
    test_loss = criterion(test_outputs, y_test_tensor_expanded)
    print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions
model.eval()
with torch.no_grad():
    src_mask = None  # Implement masking if needed
    test_predictions = model(X_test_tensor, src_mask).cpu().numpy()

# Reshape predictions to match original shape
test_predictions = test_predictions.squeeze(-1)

# Reshape y_test_seq to 2D if it's 3D
if len(y_test_seq.shape) == 3:
    y_test_seq_2d = y_test_seq.reshape(-1, y_test_seq.shape[-1])
else:
    y_test_seq_2d = y_test_seq

# Inverse transform the predictions and true values
test_predictions_original = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).reshape(test_predictions.shape)
y_test_original = scaler_y.inverse_transform(y_test_seq_2d).reshape(y_test_seq.shape)

# Ensure both arrays have the same shape
if test_predictions_original.shape != y_test_original.shape:
    if len(test_predictions_original.shape) == 2 and len(y_test_original.shape) == 3:
        y_test_original = y_test_original.squeeze(-1)
    elif len(test_predictions_original.shape) == 3 and len(y_test_original.shape) == 2:
        test_predictions_original = test_predictions_original.squeeze(-1)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(test_predictions_original - y_test_original))
print(f'Mean Absolute Error: {mae:.2f} degrees')

# Sample prediction
sample_idx = np.random.randint(0, len(test_predictions_original))
print(f'Sample Prediction: {test_predictions_original[sample_idx, -1]:.2f} degrees')
print(f'Sample True Value: {y_test_original[sample_idx, -1]:.2f} degrees')