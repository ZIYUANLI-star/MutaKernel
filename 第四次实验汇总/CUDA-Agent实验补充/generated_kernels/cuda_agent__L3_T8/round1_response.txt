import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Create transformer with batch_first=True
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            p = self.patch_size
            
            # Patch creation - use the same method as reference
            x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
            x = self.patch_to_embedding(x)
            
            # Add cls token and positional embedding
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding
            x = self.dropout(x)
            
            # Transformer
            x = self.transformer(x)
            
            # Extract cls token
            x = self.to_cls_token(x[:, 0])
            
            # MLP head - use the module directly for exact behavior match
            x = self.mlp_head(x)
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32