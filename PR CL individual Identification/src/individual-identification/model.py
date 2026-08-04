"""Model responsibilities: preparation for finetuning."""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
import timm

class EfficientProbing(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 32,
        d_out: int = 1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.d_out = d_out
        self.num_queries = num_queries
        
        # In the paper, EP-Identity uses values as they are. 
        # The repo implementation often uses a projection to enable subspace splitting.
        self.v = nn.Linear(dim, dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, cls=None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        
        # Value projection and splitting into query-specific subspaces
        v = (self.v(x).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        # Aggregate
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        
        return x_cls, attn


class EPWithClassifier(nn.Module):
    """
    Lightweight model for training on pre-extracted backbone features.
    Contains only EfficientProbing + optional projection head + classifier.
    """
    def __init__(self, in_dim, num_queries=32, out_dim=None, num_classes=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        # Efficient Probing
        self.ep = EfficientProbing(in_dim, num_queries=num_queries)
        
        # Optional projection head
        if out_dim is not None and out_dim != in_dim:
            self.head = nn.Linear(in_dim, out_dim)
        else:
            self.head = nn.Identity()
        
        # Classifier head for Identity Loss
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(self.out_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, tokens):
        """
        Forward pass on pre-extracted tokens.
        
        Args:
            tokens: Pre-extracted backbone tokens of shape (B, N, D)
        
        Returns:
            If training and classifier exists: (embeddings, logits)
            Otherwise: embeddings
        """
        # Efficient Probing pooling
        pooled, _ = self.ep(tokens)
        
        # Optional projection
        embeddings = self.head(pooled)
        
        # Return logits during training if classifier exists
        if self.training and self.classifier is not None:
            logits = self.classifier(embeddings)
            return embeddings, logits
        
        return embeddings


def prepare_model_for_finetuning_unfreeze_percentage(model, num_layers_to_unfreeze=None, trainable_percentage=None):
    """
    Freeze layers for fine-tuning, targeting a percentage of trainable parameters.

    Args:
        model: The timm model.
        num_layers_to_unfreeze (int, optional): Number of last layers to unfreeze. Used if trainable_percentage is None.
        trainable_percentage (float, optional): Target percentage of trainable parameters. Unfreezes layers from the end until this is met.

    Returns:
        Model with layers configured for fine-tuning.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Parameters that should almost always be trained during fine-tuning
    if hasattr(model, 'norm') and model.norm is not None:
        for param in model.norm.parameters():
            param.requires_grad = True
    if hasattr(model, 'norm_pre') and model.norm_pre is not None:
        for param in model.norm_pre.parameters():
            param.requires_grad = True
    if hasattr(model, 'head') and model.head is not None:
        for param in model.head.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())

    if trainable_percentage is not None:
        # Fine-grained unfreezing using leaf modules
        leaf_modules = [m for m in model.modules() if not list(m.children()) and list(m.parameters(recurse=False))]

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Targeting ~{trainable_percentage:.2f}% trainable parameters by unfreezing leaf modules.")

        # Unfreeze leaf modules from the end of the model (reverse order of definition)
        for module in reversed(leaf_modules):
            # Stop if we've reached our target
            current_percentage = (trainable_params / total_params) * 100
            if current_percentage >= trainable_percentage:
                print(f"Reached target. Stopping at {current_percentage:.2f}%.")
                break

            # Check if module parameters are already trainable (e.g. part of head/norm)
            if all(p.requires_grad for p in module.parameters(recurse=False)):
                continue

            # Unfreeze all parameters in this leaf module
            for param in module.parameters(recurse=False):
                if not param.requires_grad:
                    param.requires_grad = True
                    trainable_params += param.numel()
    elif num_layers_to_unfreeze is not None:
        # Coarse unfreezing using major blocks
        layers_to_unfreeze = []
        if hasattr(model, 'blocks'):
            layers_to_unfreeze = list(model.blocks)
        elif hasattr(model, 'stages'):
            layers_to_unfreeze = list(model.stages)
        elif hasattr(model, 'layers'):
            layers_to_unfreeze = list(model.layers)

        if layers_to_unfreeze:
            print(f"Unfreezing the last {num_layers_to_unfreeze} layer(s)/block(s).")
            for layer in layers_to_unfreeze[-num_layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Could not find standard blocks/stages/layers. No layers unfrozen by num_layers_to_unfreeze.")
    else:
        print("No fine-tuning strategy specified. Only unfreezing head/norm layers.")

    # Final count and report
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = (trainable_params / total_params) * 100
    print(f"Fine-tuning: {trainable_params:,} trainable parameters out of {total_params:,} total ({percentage:.2f}%)")

    return model


def prepare_model_for_finetuning_with_linear_head(model, embedding_dim: int | None = None):
    class ModelWithHead(nn.Module):
        def __init__(self, backbone, out_dim: int | None = None):
            super().__init__()
            self.backbone = backbone
            in_dim = getattr(backbone, "num_features", None)
            
            if in_dim is None:
                raise ValueError("Backbone does not have a num_features attribute")

            in_dim = int(in_dim)

            if out_dim is None:
                out_dim = in_dim

            self.head = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    for param in model.parameters():
        param.requires_grad = False

    # Ensure head parameters are trainable if model has one (usually replaced later though)
    if hasattr(model, 'head') and model.head is not None:
        for param in model.head.parameters():
            param.requires_grad = True

    return ModelWithHead(model, out_dim=embedding_dim)


def prepare_model_for_finetuning_with_efficient_probing(model, num_queries=32, embedding_dim: int | None = None, num_classes: int | None = None, include_cls: bool = False):
    class ModelWithEfficientProbing(nn.Module):
        def __init__(self, backbone, num_queries=32, out_dim: int | None = None, num_classes: int | None = None, include_cls: bool = False):
            super().__init__()
            self.backbone = backbone
            self.include_cls = include_cls
            in_dim = getattr(backbone, "num_features", None)
            
            if in_dim is None:
                in_dim = getattr(backbone, "embed_dim", None)

            if in_dim is None:
                raise ValueError("Backbone parameters 'num_features' or 'embed_dim' not found.")
            
            self.in_dim = int(in_dim)
            self.out_dim = out_dim if out_dim is not None else self.in_dim
            
            self.ep = EfficientProbing(self.in_dim, num_queries=num_queries)
            
            if out_dim is not None and out_dim != self.in_dim:
                 self.head = nn.Linear(self.in_dim, out_dim)
            else:
                 self.head = nn.Identity()

            self.num_classes = num_classes
            if self.num_classes is not None:
                self.classifier = nn.Linear(self.out_dim, self.num_classes)
            else:
                self.classifier = None

        def forward(self, x):
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
            else:
                 features = self.backbone(x)

            if len(features.shape) == 2:
                  raise ValueError(f"Expected 3D features (B, N, C), got {features.shape}. Ensure backbone returns patch tokens.")
            
            # Logic from README: "Default usage: pass patch tokens only. 
            # To include [CLS] among values, pass all tokens instead."
            if not self.include_cls and hasattr(self.backbone, 'num_prefix_tokens') and self.backbone.num_prefix_tokens > 0:
                 patch_tokens = features[:, self.backbone.num_prefix_tokens:, :]
            else:
                 patch_tokens = features
            
            pooled, _ = self.ep(patch_tokens)
            embeddings = self.head(pooled)
            
            if self.training and self.classifier is not None:
                logits = self.classifier(embeddings)
                return embeddings, logits
            
            return embeddings

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
        
    return ModelWithEfficientProbing(model, num_queries=num_queries, out_dim=embedding_dim, num_classes=num_classes, include_cls=include_cls)
