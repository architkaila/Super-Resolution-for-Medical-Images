## Library Imports
import math
import torch
import torch.nn.functional as F


def gaussian(window_size, sigma):
    """
    Creates a 1D Gaussian window

    Args:
        window_size: Size of the window
        sigma: Standard deviation of the Gaussian function

    Returns:
        gauss: A 1D Gaussian window
    """
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window. The Gaussian window is a 2D window that is used to weight the 
    pixels of the image during the calculation of the Structural Similarity Index (SSIM). Created 
    using a 1D Gaussian function with a specified window size and standard deviation. 

    Gaussian window focuses on the central pixels of the image and reduces the contribution of the pixels 
    away from the center (similar to human eye which is more sensitive to changes in intensity in the center
    of the image as compared to the outer regions)

    Args:
        window_size: Size of the window
        channel: Number of channels
    
    Returns:
        window: A 2D Gaussian window
    """

    ## Create a 1D Gaussian window
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    ## Create a 2D Gaussian window
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    ## Expand the 2D Gaussian window to the number of channels
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Calculate the Structural Similarity Index (SSIM) between two images

    Args:
        img1: First image
        img2: Second image
        window: Sliding window
        window_size: Size of the sliding window
        channel: Number of channels
        size_average: If True, the losses are averaged over observations for each minibatch.

    Returns:
        SSIM between img1 and img2
    """

    ## Calculate the mean of the two images
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    ## Calculate the square of mean of the two images
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    ## Calculate the product of mean of the two images
    mu1_mu2 = mu1 * mu2

    ## Calculate the variance of the two images
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ## Define the constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ## Calculate the SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ## Check if we need to average the SSIM
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Driver Function to return Structural Similarity Index (SSIM) between two images

    Args:
        img1: First image
        img2: Second image
        window_size: Size of the sliding window
        size_average: If True, the losses are averaged over observations for each minibatch.

    Returns:
        SSIM between img1 and img2
    """

    ##  Get the dimensions of the image
    (_, channel, _, _) = img1.size()

    ## Create the sliding window to applylocal smoothing operation to the images
    # intensity values of each pixel are replaced by a weighted average of the intensity values of its neighboring pixels
    window = create_window(window_size, channel)
    
    ## Check if we have a GPU available
    if img1.is_cuda:
        ## Move the window and image to the GPU
        window = window.cuda(img1.get_device())
    
    ## Convert the window to the same type as the image
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)