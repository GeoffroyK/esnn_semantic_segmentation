import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate

class UNetConvolutionBlock(nn.Module):
    '''
    Convolutional block of the model, with (Conv1 -> I&F Neuron -> Conv2 -> I&F Neuron).
    This block is used in the encoder and decoder parts of the UNet.
    '''
    def __init__(self, in_channels, out_channels):
        super(UNetConvolutionBlock, self).__init__()
      
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid())
        )

    def forward(self, x):
        return self.double_conv(x)
    

class UNetDownsample(nn.Module):
    '''
    Downsampling block: Maxpool follewed by the convolutional block.
    '''
    def __init__(self, in_channels, out_channels):
        super(UNetDownsample, self).__init__()
       
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvolutionBlock(in_channels, out_channels),
        )
    
    def forward(self, x):
        return self.down(x)
    

class UNetUpsample(nn.Module):
    '''
    Upsampling block: Upsample (or transposed conv) + skip connection, followed by the convolutional block.
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNetUpsample, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetConvolutionBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNetConvolutionBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the output of the previous layer (upsampled)
        # x2 from corresponding encoder layer (skip connection)
        
        x1 = self.up(x1)

        # Pad if needeed (if x1 does not match x2) due to rouning errors
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # concatanete along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class UNetOutConv(nn.Module):
    '''
    Final convolution layer to map the output to the desired number of classes.
    '''
    def __init__(self, in_channels, out_channels):
        super(UNetOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SpikingUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SpikingUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        # Encoder : (downsampling path)
        self.inc = UNetConvolutionBlock(n_channels, 64)             # Input: (B, 3, 256, 256) → (B, 64, 256, 256)
        self.down1 = UNetDownsample(64, 128)                        # → (B, 128, 128, 128)
        self.down2 = UNetDownsample(128, 256)                       # → (B, 256, 64, 64)
        self.down3 = UNetDownsample(256, 512)                       # → (B, 512, 32, 32)
        factor = 2 if bilinear else 1
        self.down4 = UNetDownsample(512, 1024 // factor)            # → (B, 1024, 16, 16)

        # Decoder : (upsampling path)
        self.up1 = UNetUpsample(1024, 512 // factor, bilinear)      # (1024 → 512) → (B, 512, 32, 32)
        self.up2 = UNetUpsample(512, 256 // factor, bilinear)       # → (B, 256, 64, 64)
        self.up3 = UNetUpsample(256, 128 // factor, bilinear)       # → (B, 128, 128, 128)
        self.up4 = UNetUpsample(128, 64, bilinear)                  # → (B, 64, 256, 256)

        # Output
        self.outc = UNetOutConv(64, n_classes)                      # → (B, n_classes, 256, 256)

    def forward(self, x):
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