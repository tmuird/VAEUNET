import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.attention = AttentionGate(in_channels, skip_channels, in_channels//4)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            skip = self.attention(x, skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetResNet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone='resnet34', pretrained=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load pretrained backbone
        self.encoder = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=n_channels
        )
        encoder_channels = self.encoder.feature_info.channels()
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(encoder_channels[-1], encoder_channels[-2], 512),
            DecoderBlock(512, encoder_channels[-3], 256),
            DecoderBlock(256, encoder_channels[-4], 128),
            DecoderBlock(128, encoder_channels[0], 64)
        ])
        
        # Final conv
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Deep supervision
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(512, n_classes, 1),
            nn.Conv2d(256, n_classes, 1),
            nn.Conv2d(128, n_classes, 1)
        ])
        
    def forward(self, x):
        # Get original input size
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = features[-1]
        deep_outputs = []
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i+2)] if i < len(features)-1 else None
            x = decoder_block(x, skip)
            if i < 3:  # Don't add deep supervision for last block
                deep_out = self.deep_supervision[i](x)
                deep_out = F.interpolate(deep_out, size=input_size, mode='bilinear', align_corners=True)
                deep_outputs.append(deep_out)
        
        # Final conv and ensure output size matches input size
        output = self.final_conv(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training and len(deep_outputs) > 0:
            return [output] + deep_outputs
        return output
