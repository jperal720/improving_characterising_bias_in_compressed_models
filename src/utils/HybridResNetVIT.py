import torch
import torch.nn as nn

class VITEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        return x
    
class HybridResNetVIT(nn.Module):
    def __init__(self, resnet, num_layers=2, embed_dim=512, num_heads=8, classes=1):
        super().__init__()

        self.resnet = resnet
        #Identity to "remove" the final fc layer
        self.resnet.fc = nn.Identity()

        # VIT Encoder Block(s) (change num_layers=1 for single multi-headed block)
        self.vit_encoder = nn.Sequential(
            *[VITEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
              for _ in range(num_layers)]
        )

        self.fc = nn.Linear(embed_dim, classes)

    def forward(self, x):
        # ResNet pass
        features = self.resnet(x)
        features = features.unsqueeze(0)

        # VIT pass
        vit_output = self.vit_encoder(features)
        vit_output = vit_output.squeeze(0)

        x = self.fc(vit_output)
        return x

    

