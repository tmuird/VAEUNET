""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.utils.checkpoint as checkpoint


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_checkpointing=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_checkpointing = use_checkpointing

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if self.use_checkpointing and self.training:
            def up1_func(x5, x4):
                return self.up1(x5, x4)
            
            def up2_func(x, x3):
                return self.up2(x, x3)
            
            def up3_func(x, x2):
                return self.up3(x, x2)
            
            def up4_func(x, x1):
                return self.up4(x, x1)

            x1 = checkpoint.checkpoint(self.inc, x, use_reentrant=False)
            x2 = checkpoint.checkpoint(self.down1, x1, use_reentrant=False)
            x3 = checkpoint.checkpoint(self.down2, x2, use_reentrant=False)
            x4 = checkpoint.checkpoint(self.down3, x3, use_reentrant=False)
            x5 = checkpoint.checkpoint(self.down4, x4, use_reentrant=False)
            x = checkpoint.checkpoint(up1_func, x5, x4, use_reentrant=False)
            x = checkpoint.checkpoint(up2_func, x, x3, use_reentrant=False)
            x = checkpoint.checkpoint(up3_func, x, x2, use_reentrant=False)
            x = checkpoint.checkpoint(up4_func, x, x1, use_reentrant=False)
            logits = checkpoint.checkpoint(self.outc, x, use_reentrant=False)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        return logits

    def enable_checkpointing(self):
        """Enable gradient checkpointing"""
        self.use_checkpointing = True

    def disable_checkpointing(self):
        """Disable gradient checkpointing"""
        self.use_checkpointing = False