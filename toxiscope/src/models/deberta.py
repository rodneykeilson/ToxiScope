"""
ToxiScope Model Architecture: DeBERTa-v3 for Toxicity Classification

DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention) is state-of-the-art
for English NLU tasks including toxicity detection on Jigsaw benchmarks.

Key innovations over BERT/RoBERTa:
1. Disentangled attention - separate content and position encodings
2. Enhanced mask decoder - better position information in MLM
3. ELECTRA-style pre-training (replaced token detection)

References:
- He et al. (2021) "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training"
- He et al. (2020) "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
- Jigsaw Toxic Comment Classification Challenge (Kaggle)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
from transformers import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    DebertaV2Config,
    AutoModel,
    AutoConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class ToxiScopeConfig:
    """Configuration for ToxiScope model enhancements."""
    
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 7
    pooling_strategy: Literal["cls", "mean", "attention"] = "cls"
    use_multi_sample_dropout: bool = True
    dropout_samples: int = 5
    classifier_dropout: float = 0.2
    hidden_size: int = 768  # Will be overridden from pretrained config
    

class AttentionPooling(nn.Module):
    """
    Learned attention pooling over sequence.
    
    Instead of using [CLS] token or mean pooling, learns attention weights
    over all tokens to create a weighted representation.
    
    Reference: Yang et al. (2016) "Hierarchical Attention Networks"
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            pooled: (batch_size, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax to get weights
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (hidden_states * weights).sum(dim=1)  # (batch, hidden_size)
        
        return pooled


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout for regularization.
    
    Applies dropout multiple times and averages the results during training.
    Improves generalization, especially for imbalanced datasets.
    
    Reference: Inoue (2019) "Multi-Sample Dropout for Accelerated Training"
    """
    
    def __init__(self, classifier: nn.Module, dropout_rate: float, num_samples: int = 5):
        super().__init__()
        self.classifier = classifier
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_samples)
        ])
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor to classifier
            
        Returns:
            Averaged logits from multiple dropout samples
        """
        if self.training:
            # Average over multiple dropout samples during training
            outputs = torch.stack([
                self.classifier(dropout(x)) for dropout in self.dropouts
            ], dim=0)
            return outputs.mean(dim=0)
        else:
            # Single forward pass during inference
            return self.classifier(x)


class ToxiScopeModel(nn.Module):
    """
    ToxiScope: DeBERTa-v3 with enhancements for toxicity classification.
    
    Enhancements over vanilla DeBERTa:
    1. Flexible pooling (CLS, mean, attention)
    2. Multi-sample dropout for regularization
    3. Optimized for multi-label classification
    
    Supports: DeBERTa-v3-base, DeBERTa-v3-large, RoBERTa variants
    """
    
    def __init__(self, config: ToxiScopeConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained encoder
        self.encoder_config = AutoConfig.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        # Update hidden size from pretrained config
        hidden_size = self.encoder_config.hidden_size
        config.hidden_size = hidden_size
        
        # Pooling layer
        if config.pooling_strategy == "attention":
            self.pooler = AttentionPooling(hidden_size, config.classifier_dropout)
        else:
            self.pooler = None
        
        # Classifier head
        classifier = nn.Linear(hidden_size, config.num_labels)
        
        # Wrap with multi-sample dropout if enabled
        if config.use_multi_sample_dropout:
            self.classifier = MultiSampleDropout(
                classifier, 
                config.classifier_dropout,
                config.dropout_samples
            )
        else:
            self.dropout = nn.Dropout(config.classifier_dropout)
            self.classifier = classifier
        
        # Initialize classifier weights
        self._init_weights(classifier)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _pool(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply pooling strategy to get sequence representation."""
        
        if self.config.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == "mean":
            # Mean pooling over non-padding tokens
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1e-9)
                return sum_hidden / count
            else:
                return hidden_states.mean(dim=1)
        
        elif self.config.pooling_strategy == "attention":
            return self.pooler(hidden_states, attention_mask)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> SequenceClassifierOutput:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs (batch_size, seq_len) - optional
            labels: Ground truth labels (batch_size, num_labels) - optional
            return_dict: Whether to return dict output
            
        Returns:
            SequenceClassifierOutput with logits and optional loss
        """
        # Encode
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Only pass token_type_ids if model supports it (DeBERTa-v3 doesn't use them)
        if token_type_ids is not None and hasattr(self.encoder_config, 'type_vocab_size'):
            if self.encoder_config.type_vocab_size > 0:
                encoder_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Pool
        pooled = self._pool(hidden_states, attention_mask)
        
        # Classify
        if self.config.use_multi_sample_dropout:
            logits = self.classifier(pooled)
        else:
            logits = self.classifier(self.dropout(pooled))
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        if return_dict:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            )
        else:
            return (loss, logits) if loss is not None else logits
    
    def freeze_encoder(self, num_layers_to_freeze: int = 0):
        """
        Freeze encoder layers for gradual unfreezing.
        
        Args:
            num_layers_to_freeze: Number of layers to freeze from bottom.
                                  0 = train all, -1 = freeze all
        """
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = (num_layers_to_freeze == 0)
        
        # Freeze encoder layers
        if hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'layer'):
            layers = self.encoder.layer
        else:
            return
        
        if num_layers_to_freeze == -1:
            # Freeze all
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            # Freeze first N layers
            for i, layer in enumerate(layers):
                freeze = i < num_layers_to_freeze
                for param in layer.parameters():
                    param.requires_grad = not freeze
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_toxiscope_model(
    model_name: str = "microsoft/deberta-v3-base",
    num_labels: int = 7,
    pooling_strategy: str = "cls",
    use_multi_sample_dropout: bool = True,
    dropout_samples: int = 5,
    classifier_dropout: float = 0.2,
) -> ToxiScopeModel:
    """
    Factory function to create ToxiScope model.
    
    Args:
        model_name: HuggingFace model identifier
            - "microsoft/deberta-v3-base" (recommended)
            - "microsoft/deberta-v3-large"
            - "roberta-base"
            - "roberta-large"
        num_labels: Number of toxicity labels
        pooling_strategy: How to pool encoder outputs
        use_multi_sample_dropout: Whether to use multi-sample dropout
        dropout_samples: Number of dropout samples
        classifier_dropout: Dropout rate for classifier
        
    Returns:
        Configured ToxiScopeModel
    """
    config = ToxiScopeConfig(
        model_name=model_name,
        num_labels=num_labels,
        pooling_strategy=pooling_strategy,
        use_multi_sample_dropout=use_multi_sample_dropout,
        dropout_samples=dropout_samples,
        classifier_dropout=classifier_dropout,
    )
    
    model = ToxiScopeModel(config)
    
    print(f"Created ToxiScope model:")
    print(f"  Base: {model_name}")
    print(f"  Pooling: {pooling_strategy}")
    print(f"  Multi-sample dropout: {use_multi_sample_dropout} ({dropout_samples} samples)")
    print(f"  Total params: {model.get_total_params():,}")
    print(f"  Trainable params: {model.get_trainable_params():,}")
    
    return model


# For HuggingFace Trainer compatibility
class ToxiScopeForSequenceClassification(DebertaV2PreTrainedModel):
    """
    HuggingFace-compatible wrapper for ToxiScope.
    
    This class inherits from DebertaV2PreTrainedModel to enable
    seamless integration with HuggingFace Trainer.
    """
    
    def __init__(self, config: DebertaV2Config, toxiscope_config: Optional[ToxiScopeConfig] = None):
        super().__init__(config)
        
        if toxiscope_config is None:
            toxiscope_config = ToxiScopeConfig(
                model_name=config.name_or_path if hasattr(config, 'name_or_path') else "microsoft/deberta-v3-base",
                num_labels=config.num_labels,
            )
        
        self.deberta = DebertaV2Model(config)
        self.toxiscope_config = toxiscope_config
        
        # Pooling
        if toxiscope_config.pooling_strategy == "attention":
            self.pooler = AttentionPooling(config.hidden_size, toxiscope_config.classifier_dropout)
        else:
            self.pooler = None
        
        # Classifier with optional multi-sample dropout
        classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        if toxiscope_config.use_multi_sample_dropout:
            self.classifier = MultiSampleDropout(
                classifier,
                toxiscope_config.classifier_dropout,
                toxiscope_config.dropout_samples
            )
            self.dropout = None
        else:
            self.dropout = nn.Dropout(toxiscope_config.classifier_dropout)
            self.classifier = classifier
        
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Pooling
        if self.toxiscope_config.pooling_strategy == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.toxiscope_config.pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(1)
        elif self.toxiscope_config.pooling_strategy == "attention":
            pooled = self.pooler(hidden_states, attention_mask)
        else:
            pooled = hidden_states[:, 0, :]
        
        # Classify
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    # Test model creation
    print("Testing ToxiScope DeBERTa model creation...")
    
    # Test with different configurations
    configs = [
        ("microsoft/deberta-v3-base", "cls", True),
        ("microsoft/deberta-v3-base", "mean", False),
        ("microsoft/deberta-v3-base", "attention", True),
    ]
    
    for model_name, pooling, msd in configs:
        print(f"\n{'='*60}")
        model = create_toxiscope_model(
            model_name=model_name,
            pooling_strategy=pooling,
            use_multi_sample_dropout=msd,
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 128
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
        dummy_mask = torch.ones(batch_size, seq_len)
        dummy_labels = torch.randint(0, 2, (batch_size, 7)).float()
        
        output = model(
            input_ids=dummy_input,
            attention_mask=dummy_mask,
            labels=dummy_labels,
        )
        
        print(f"  Loss: {output.loss.item():.4f}")
        print(f"  Logits shape: {output.logits.shape}")
    
    print("\n✅ All tests passed!")
