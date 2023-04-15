import torch
from torch import nn
from torchvision.models import mobilenet_v2

class GeneratorLoss(nn.Module):
    """
    This class defines the Generator loss function.
    """
    def __init__(self):
        """
        Initialize the GeneratorLoss class.
        """
        super(GeneratorLoss, self).__init__()

        ## Initialize the mobile net v2 model to extract features for perceptual loss
        vgg = mobilenet_v2(pretrained=True)

        ## Get the feature extraction layers of the network and set the network to evaluation mode
        loss_network = nn.Sequential(*list(vgg.features)).eval()

        ## Freeze the weights of the loss network
        for param in loss_network.parameters():
            param.requires_grad = False

        ## Save the perceptual loss estimator network    
        self.loss_network = loss_network

        ## Define the MSE loss object
        self.mse_loss = nn.MSELoss()

        ## Define the TV loss object
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        """
        Forward pass of the loss function for the generator. The total perceptual loss function is 
        formulated as the weighted sum of adversarial loss, content loss, naive image loss and TV loss.

        Args:
            out_labels (torch.Tensor): The output of the discriminator for the generated images
            out_images (torch.Tensor): The generated images by the generator network
            target_images (torch.Tensor): The actual target images

        Returns:
            total_loss (torch.Tensor): The total loss for the generator
        """

        ## Calculate the Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        ## Calculate the Perception/Content Loss (Eucleadian distance b/w generated & actual image feature maps)
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        ## Calculate Naive Image Loss (mse between the generated & actual images, (pixel diff mse))
        image_loss = self.mse_loss(out_images, target_images)
        
        ## Calculate the TV Loss
        tv_loss = self.tv_loss(out_images)

        total_loss = image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

        return total_loss


class TVLoss(nn.Module):
    """
    A class that defines the Total Variation loss. TV loss measures the variation or changes in 
    intensity or color between adjacent pixels in an image. TV loss is defined as the sum of the 
    absolute differences between neighboring pixels in both the horizontal and vertical directions 
    in an image
    """
    def __init__(self, tv_loss_weight=1):
        """
        Constructor to initialize the TVLoss class.

        Args:
            tv_loss_weight (float): The weight of the TV loss. Default: 1

        Returns:
            None
        """
        super(TVLoss, self).__init__()

        ## Initialize the TV loss weight
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        """
        Forward pass of the TV loss function.

        Args:
            x (torch.Tensor): The input tensor for which the TV loss is to be calculated
        
        Returns:
            tv_loss (torch.Tensor): The total variation loss
        """

        ## Get the batch size, height and width of the input images tensor
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        ## Calculate total number of values in the height and width dimensions of the input tensor
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        ## Total variation along the height dimension of the input tenso
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        ## Total variation along the width dimension of the input tensor
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        ## Calculate the total variation loss per sample
        tv_loss = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return tv_loss

    @staticmethod
    def tensor_size(t):
        """
        A static method to calculate the size of the input tensor. (excluding the batch size)

        Args:
            t (torch.Tensor): The input tensor for which the size is to be calculated

        Returns:
            size (int): The size of the input tensor
        """
        return t.size()[1] * t.size()[2] * t.size()[3]