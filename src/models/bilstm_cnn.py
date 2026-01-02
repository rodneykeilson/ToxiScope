"""
ToxiScope: BiLSTM-CNN Baseline Model

Non-transformer baseline for ablation study comparison.
Combines bidirectional LSTM with CNN for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class BiLSTM_CNN(nn.Module):
    """
    BiLSTM-CNN architecture for multilabel text classification.
    
    Architecture:
    1. Embedding layer (optionally pretrained)
    2. Bidirectional LSTM
    3. Multi-kernel CNN
    4. Concatenation + FC layers
    5. Multilabel classification head
    
    This serves as a strong non-transformer baseline for comparison
    with DeBERTa-based models.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        cnn_filters: int = 256,
        cnn_kernel_sizes: List[int] = [3, 4, 5],
        fc_hidden_size: int = 512,
        num_labels: int = 7,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_hidden_size: Hidden size of LSTM
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: Dropout between LSTM layers
            cnn_filters: Number of filters per CNN kernel
            cnn_kernel_sizes: List of kernel sizes for CNN
            fc_hidden_size: Hidden size of FC layers
            num_labels: Number of output labels
            dropout: Dropout rate for classification head
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embeddings
            padding_idx: Index used for padding
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_labels = num_labels
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
        )
        
        # CNN layers with multiple kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=lstm_hidden_size * 2,  # Bidirectional
                out_channels=cnn_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in cnn_kernel_sizes
        ])
        
        # Calculate total features after CNN
        total_cnn_features = cnn_filters * len(cnn_kernel_sizes)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(total_cnn_features + lstm_hidden_size * 2, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size // 2, num_labels),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) - token indices
            attention_mask: (batch, seq_len) - mask for padding
            labels: (batch, num_labels) - multilabel targets
        
        Returns:
            Dict with 'loss', 'logits'
        """
        batch_size = input_ids.size(0)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Pack sequences for LSTM if mask provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # lstm_out: (batch, seq_len, lstm_hidden * 2)
        
        # Get final hidden state for LSTM path
        # Concatenate forward and backward final hidden states
        hidden_forward = hidden[-2, :, :]  # Last layer forward
        hidden_backward = hidden[-1, :, :]  # Last layer backward
        lstm_features = torch.cat([hidden_forward, hidden_backward], dim=-1)
        
        # CNN path
        # Transpose for conv1d: (batch, channels, seq_len)
        cnn_input = lstm_out.transpose(1, 2)
        
        # Apply each conv + max pooling
        cnn_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(cnn_input))  # (batch, filters, seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            cnn_outputs.append(pooled)
        
        # Concatenate CNN features
        cnn_features = torch.cat(cnn_outputs, dim=-1)
        
        # Combine LSTM and CNN features
        combined = torch.cat([lstm_features, cnn_features], dim=-1)
        
        # Classification
        logits = self.fc(combined)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        return {
            "loss": loss,
            "logits": logits,
            "lstm_features": lstm_features,
            "cnn_features": cnn_features,
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        thresholds: Optional[Dict[str, float]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Make predictions with threshold application."""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)
        
        if label_names is None:
            label_names = [f"label_{i}" for i in range(self.num_labels)]
        
        if thresholds is None:
            thresholds = {label: 0.5 for label in label_names}
        
        threshold_tensor = torch.tensor([
            thresholds.get(label, 0.5) for label in label_names
        ]).to(probs.device)
        
        predictions = (probs >= threshold_tensor).int()
        
        return {
            "probabilities": probs,
            "predictions": predictions,
            "logits": logits,
        }


class TextCNN(nn.Module):
    """
    TextCNN model for comparison.
    
    Simpler architecture using only convolutional layers.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 256,
        kernel_sizes: List[int] = [2, 3, 4, 5],
        num_labels: int = 7,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        concat = torch.cat(conv_outputs, dim=-1)
        dropped = self.dropout(concat)
        logits = self.fc(dropped)
        
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return {"loss": loss, "logits": logits}


def create_bilstm_cnn(
    vocab_size: int,
    config: Optional[Dict] = None,
) -> BiLSTM_CNN:
    """Factory function to create BiLSTM-CNN model."""
    default_config = {
        "embedding_dim": 300,
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.3,
        "cnn_filters": 256,
        "cnn_kernel_sizes": [3, 4, 5],
        "fc_hidden_size": 512,
        "num_labels": 7,
        "dropout": 0.5,
    }
    
    if config:
        default_config.update(config)
    
    return BiLSTM_CNN(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Test model
    print("Testing BiLSTM-CNN model...")
    
    vocab_size = 50000
    batch_size = 4
    seq_len = 100
    
    model = create_bilstm_cnn(vocab_size)
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size, 7)).float()
    
    outputs = model(input_ids, attention_mask, labels)
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
