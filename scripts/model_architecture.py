import torch
from torch import nn


class SeperableConv2d(nn.Module):
    """
    A class that defines the Depthwise Seperable Convolutional Layer. It is a two step operation: 
    (1) The depth-wise Convolution is applied to a single image channel at a time, 
    followed by the 
    (2) Point-wise Convolution, which is a 1x1 Convolution operation performed on M image channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        """
        Constructor method for the SeperableConv2d class.

        Args:
            in_channels (int): number of input image channels.
            out_channels (int): number of output image channels.
            kernel_size (int): size of the convolutional kernel.
            stride (int): stride of the convolution.
            padding (int): padding to be applied to the input image.
            bias (bool): whether to use bias or not.

        Returns:
            None
        """
        super(SeperableConv2d, self).__init__()

        ## Initialize the required variables

        ## Define Depthwise Convolution
        # Each input chanel is convolved with a different set of filters,
        self.depthwise = nn.Conv2d(
            in_channels, # Number of input channels
            in_channels, # Number of output channels same as input channels
            kernel_size=kernel_size,
            stride = stride,
            groups=in_channels, # At groups = in_channels, each input channel is convolved with its own set of filters
            bias=bias,
            padding=padding
        )

        ## Define Pointwise Convolution
        # 1x1 convolution is applied to linearly combine the outputs from the depthwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        """
        Function to perform the forward pass of the Depthwise Seperable Conv2d layer. It
        first applies the Depthwise Convolution and then the Pointwise Convolution.

        Args:
            x (Tensor): input tensor of shape (N, C, H, W).
        
        Returns:
            x (Tensor): output tensor of shape (N, C, H, W).
        """

        return self.pointwise(self.depthwise(x))
    

    
class ConvBlock(nn.Module):
    """
    A class that defines the Convolutional Block. It consists of a Depthwise Convolutional Layer,
    Bath Normalization and a Leaky ReLU/PRelu for activation.
    """
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=True, discriminator=False, **kwargs):
        """
        Constructor method for the ConvBlock class.

        Args:
            in_channels (int): number of input image channels.
            out_channels (int): number of output image channels.
            use_act (bool): whether to use activation or not.
            use_bn (bool): whether to use batch normalization or not.
            discriminator (bool): whether to use Leaky ReLU or PReLU.
            **kwargs: keyword arguments for the Depthwise Seperable Convolutional Layer.

        Returns:
            None
        """
        super(ConvBlock, self).__init__()
        
        ## Initialize the required variables

        # Whether to use activation or not
        self.use_act = use_act

        # Define the Depthwise Seperable Convolutional Layer
        self.cnn = SeperableConv2d(in_channels, out_channels, **kwargs, bias=not use_bn)

        # Define Batch Normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Define Activation based on whether it is a discriminator or generator conv block
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        """
        Function to perform the forward pass of the ConvBlock.

        Args:
            x (Tensor): input tensor of shape (N, C, H, W).
        
        Returns:
            res (Tensor): output tensor of shape (N, C, H, W).
        """

        ## if use_act is True, apply activation after batch normalization
        if self.use_act:
            res = self.act(self.bn(self.cnn(x)))

        ## else, apply batch normalization only
        else:
            res = self.bn(self.cnn(x))

        return res


class UpsampleBlock(nn.Module):
    """
    A class that defines the Upsampling Block. It consists of a Depthwise Convolutional Layer,
    Pixel Shuffle and a PReLU for activation.
    """
    def __init__(self, in_channels, scale_factor):
        """
        Constructor method for the UpsampleBlock class.

        Args:
            in_channels (int): number of input image channels.
            scale_factor (int): scale factor for the upsampling.

        Returns:
            None
        """
        super(UpsampleBlock, self).__init__()
        
        ## Initialize the required variables

        # Define the Depthwise Seperable Convolutional Layer
        self.conv = SeperableConv2d(in_channels, in_channels * scale_factor**2, kernel_size=3, stride=1, padding=1)
        
        # Define Pixel Shuffle
        self.ps = nn.PixelShuffle(scale_factor) # (in_channels * 4, H, W) -> (in_channels, H*2, W*2)
        
        # Define PReLU activation
        self.act = nn.PReLU(num_parameters=in_channels)
    
    def forward(self, x):
        """
        Function to perform the forward pass of the UpsampleBlock.

        Args:
            x (Tensor): input tensor of shape (N, C, H, W).

        Returns:
            (Tensor): output tensor of shape (N, C, H, W).
        """
        return self.act(self.ps(self.conv(x)))
        

class ResidualBlock(nn.Module):
    """
    A class that defines the Residual Block. It consists of two Convolutional Blocks and
    a skip connection.
    """
    def __init__(self, in_channels):
        """
        Constructor method for the ResidualBlock class.

        Args:
            in_channels (int): number of input image channels.
        
        Returns:
            None
        """
        super(ResidualBlock, self).__init__()
        
        ## Initialize the required variables

        # Define the two Convolutional Block with activation
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Define the second Convolutional Block without activation
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )
        
    def forward(self, x):
        """
        Function to perform the forward pass of the ResidualBlock.

        Args:
            x (Tensor): input tensor of shape (N, C, H, W).
        
        Returns:
            out (Tensor): output tensor of shape (N, C, H, W).
        """

        ## Pass the input through the first Convolutional Block
        out = self.block1(x)

        ## Pass the output through the second Convolutional Block
        out = self.block2(out)

        return out + x ## Add the initial input to the output to perform skip connect and return
     
class Generator(nn.Module):
    """
    A class that defines the Swift SRGAN Generator architecture.
    """

    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_blocks: int = 16, upscale_factor: int = 4):
        """
        Constructor method for the Generator class.

        Args:
            in_channels (int): number of input image channels.
            num_channels (int): number of channels in the first layer.
            num_blocks (int): number of residual blocks.
            upscale_factor (int): factor by which to upscale the image.

        Returns:
            None
        """
        super(Generator, self).__init__()
        
        ## Initial ConvBlock
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        
        ## Set of Sequential Residual Blocks (16 by default)
        self.residual = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        ## Intermediate ConvBlock
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        
        ## Upsampling Blocks (2 by default)
        self.upsampler = nn.Sequential(
            *[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor//2)]
        )

        ## Final ConvBlock
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        """
        Forward pass of the Generator architecture.

        Args:
            x (torch.Tensor): input tensor.
        
        Returns:
            torch.Tensor
        """

        initial = self.initial(x) # Pass image thorugh initial ConvBlock
        x = self.residual(initial) # Pass resultant image through set of sequential residual blocks
        x = self.convblock(x) + initial # Pass resultant image through intermediate ConvBlock and add the skip connect of the initial image
        x = self.upsampler(x) # Pass resultant image through the upsampling blocks

        return (torch.tanh(self.final_conv(x)) + 1) / 2 # Pass resultant image through final ConvBlock and return the output after applying the tanh activation function


class Discriminator(nn.Module):
    """
    A class that defines the Swift SRGAN Discriminator architecture.
    """

    def __init__(self, in_channels: int = 3, features: tuple = (64, 64, 128, 128, 256, 256, 512, 512),) -> None:
        """
        Constructor method for the Discriminator class.

        Args:
            in_channels (int): number of input image channels.
            features (tuple): number of features in each layer.

        Returns:
            None
        """
        super(Discriminator, self).__init__()
        
        ## Define a list of Sequential ConvBlocks (8 by default) 
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True, # Do not use BatchNorm in the first layer
                )
            )
            in_channels = feature
        ## Initialize the Sequential ConvBlocks
        self.blocks = nn.Sequential(*blocks)

        ## Define the classifier (Final layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)), # Average Pooling
            nn.Flatten(), # Flatten the resultant image
            nn.Linear(512 * 6 * 6, 1024), # Add a Linear fully connected layer
            nn.LeakyReLU(0.2, inplace=True), # Add a LeakyReLU activation function
            nn.Linear(1024, 1), # Add a Linear fully connected layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator model.

        Args:
            x (torch.Tensor): input tensor.
        
        Returns:
            torch.Tensor
        """

        x = self.blocks(x) # Pass image through the Sequential ConvBlocks
        classify = self.classifier(x) # Pass resultant image through the fully connected classifier

        return torch.sigmoid(classify) # Return the output after applying the sigmoid activation function