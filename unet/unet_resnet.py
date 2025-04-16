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
    def __init__(self, in_channels, skip_channels, out_channels, latent_dim, use_attention=True, use_skip=True, use_latent=True):
        super().__init__()
        # Project latent vector to match spatial dimensions (only create if used)
        self.use_latent = use_latent
        if use_latent:
            self.z_proj = nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, 1),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU(inplace=True)
            )
        
        self.use_skip = use_skip
        # Attention can only be enabled if skip connections are enabled
        self.use_attention = use_attention and use_skip
        
        # Attention gate (only create if used)
        if self.use_attention:
            self.attention = AttentionGate(in_channels, skip_channels, in_channels//4)
        
        # Calculate input channels for first convolution based on what's being used
        input_channels = in_channels
        if use_skip:
            input_channels += skip_channels
        if use_latent:
            input_channels += latent_dim
            
        # Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip, z):
        # Compute output size based on skip connection if it exists
        if skip is not None:
            output_size = skip.shape[2:]
        else:
            output_size = (x.shape[2] * 2, x.shape[3] * 2)
            
        # Upsample x to match skip connection size
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=True)
        
        # Prepare components for concatenation
        components = [x]
        
        # Add skip connection if used
        if skip is not None and self.use_skip:
            # Apply attention if enabled
            if self.use_attention:
                skip = self.attention(x, skip)
            components.append(skip)
        
        # Add latent injection if used
        if self.use_latent:
            z_proj = F.interpolate(z, size=output_size, mode='bilinear', align_corners=True)
            z_proj = self.z_proj(z_proj)
            components.append(z_proj)
        
        # Concatenate all components and apply convolutions
        x = torch.cat(components, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetResNet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone='resnet34', pretrained=True, latent_dim=32, 
                 use_attention=True, use_skip=True, latent_injection='all'):
        """
        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
            backbone: ResNet backbone to use
            pretrained: Whether to use pretrained weights
            latent_dim: Latent space dimensionality
            use_attention: Whether to use attention gates
            use_skip: Whether to use skip connections
            latent_injection: Strategy for latent space injection
                - 'all': Inject at bottleneck AND all decoder levels (default)
                - 'first': Inject at bottleneck AND first decoder level
                - 'last': Inject at bottleneck AND last decoder level
                - 'bottleneck': Inject ONLY at the bottleneck (decoder starts from z)
                - 'inject_no_bottleneck': Inject at all decoder levels BUT NOT at bottleneck (decoder starts from encoder features)
                - 'none': No latent injection at all (decoder starts from encoder features)
                - list of ints: Inject at bottleneck AND specified decoder levels (0-based index)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.latent_injection = latent_injection
        
        # Load pretrained backbone
        self.encoder = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=n_channels
        )
        encoder_channels = self.encoder.feature_info.channels()
        
        # VAE heads for latent space
        self.mu_head = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], latent_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], latent_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        # Initial projection of latent vector
        self.z_initial = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Determine which decoder blocks should use latent space
        if isinstance(latent_injection, list):
            # Specific levels defined by list
            use_latent_list = [i in latent_injection for i in range(4)]
        elif latent_injection == 'all' or latent_injection == 'inject_no_bottleneck': # Added 'inject_no_bottleneck' here
            use_latent_list = [True, True, True, True]
        elif latent_injection == 'first':
            use_latent_list = [True, False, False, False]
        elif latent_injection == 'last':
            use_latent_list = [False, False, False, True]
        elif latent_injection == 'bottleneck' or latent_injection == 'none':
            use_latent_list = [False, False, False, False]
        else:
            # Default to all
            use_latent_list = [True, True, True, True]
            latent_injection = 'all' # Correct the mode if unknown
            
        # Whether to use bottleneck injection (decoder starts from z)
        # This is True for all modes EXCEPT 'none' and the new 'inject_no_bottleneck'
        self.use_bottleneck = latent_injection not in ['none', 'inject_no_bottleneck']
            
        # Decoder blocks with parameterized latent injection
        self.use_skip = use_skip
        self.use_attention = use_attention and use_skip
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(512, encoder_channels[-2], 512, latent_dim, use_attention, use_skip, use_latent_list[0]),
            DecoderBlock(512, encoder_channels[-3], 256, latent_dim, use_attention, use_skip, use_latent_list[1]),
            DecoderBlock(256, encoder_channels[-4], 128, latent_dim, use_attention, use_skip, use_latent_list[2]),
            DecoderBlock(128, encoder_channels[0], 64, latent_dim, use_attention, use_skip, use_latent_list[3])
        ])
        
        # Final conv
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # Get original input size
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)
        
        # Get latent variables from bottleneck
        x_enc = features[-1]
        mu = self.mu_head(x_enc).squeeze(-1).squeeze(-1)  # Shape: [B, latent_dim]
        logvar = self.logvar_head(x_enc).squeeze(-1).squeeze(-1)  # Shape: [B, latent_dim]
        
        # Reparameterize
        # Determine if we should sample from the latent space
        # For 'none' and 'inject_no_bottleneck', we don't need to use the sampled latent in the bottleneck
        should_sample = self.latent_injection not in ['none', 'inject_no_bottleneck']
        if should_sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu # Shape: [B, latent_dim]
        
        # Reshape z to spatial dimensions
        z = z.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
        
        # Initial feature size matches the smallest encoder feature
        initial_size = features[-1].shape[2:]
        z_spatial = F.interpolate(z, size=initial_size, mode='bilinear', align_corners=True)
        
        # Apply bottleneck processing based on injection strategy
        if self.use_bottleneck:
            # Use latent for bottleneck
            x = self.z_initial(z_spatial)
        else:
            # No latent injection at bottleneck - use encoder features directly
            x = features[-1]
        
        # Decoder with conditional z injection at each stage
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i+2)] if i < len(features)-1 and self.use_skip else None
            x = decoder_block(x, skip, z_spatial)
        
        # Final conv and ensure output size matches input size
        output = self.final_conv(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        return output, mu, logvar

    def encode(self, x):
        """Encode input to latent space, returning mean and logvar."""
        features = self.encoder(x)
        x = features[-1]
        mu = self.mu_head(x).squeeze(-1).squeeze(-1)
        logvar = self.logvar_head(x).squeeze(-1).squeeze(-1)
        return mu, logvar

    def decode(self, z, input_size=None):
        """Decode latent vector z to segmentation mask."""
        # Get encoder features for skip connection sizes
        with torch.no_grad():
            features = self.encoder(torch.zeros(1, self.n_channels, 512, 512, device=z.device))
        
        # Reshape z to spatial dimensions
        z = z.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
        initial_size = features[-1].shape[2:]
        z_spatial = F.interpolate(z, size=initial_size, mode='bilinear', align_corners=True)
        
        # Apply bottleneck processing based on injection strategy
        if self.use_bottleneck:
            # Use latent for bottleneck
            x = self.z_initial(z_spatial)
        else:
            # For 'none' mode, we still need a starting point, use encoder features with channel padding
            x = torch.zeros(z.size(0), 512, initial_size[0], initial_size[1], device=z.device)
        
        # Decode with conditional z injection
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i+2)] if i < len(features)-1 and self.use_skip else None
            x = decoder_block(x, skip, z_spatial)
        
        # Final conv and optional resize
        output = self.final_conv(x)
        if input_size is not None:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        return output

        """Sample from the latent space and generate segmentation masks.
        Args:
            num_samples: Number of samples to generate
            temp: Temperature parameter for sampling (higher = more diverse)
            device: Device to generate on
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device) * temp
        
        # Generate masks
        with torch.no_grad():
            masks = self.decode(z)
            masks = torch.sigmoid(masks)
        
        return masks