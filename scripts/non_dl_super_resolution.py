## Library imports
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor
import pickle
from tqdm import tqdm
import sys
import math
import pandas as pd

## Local imports
from model_metrics import ssim

def naive_super_resolution(image_path, lr_scale=256, hr_scale=1024):
    """
    Naive super resolution using bicubic interpolation.

    Args:
        image_path (str): Path to the image to be super resolved.
        scale_factor (int): Scale factor to be used for super resolution.
    
    Returns:
        Image: Super resolved image.
    """

    # Load an image
    hr_image = Image.open(image_path).convert('RGB')

    ## Create the LR image transformer by downsampling the HR image and applying bicubic interpolation
    lr_scale = transforms.Resize((lr_scale,lr_scale), interpolation=Image.BICUBIC)

    ## Create the restored HR image tranformer (simple classical method) by upsampling the LR image and applying bicubic interpolation
    hr_scale = transforms.Resize((hr_scale,hr_scale), interpolation=Image.BICUBIC)

    ## Create the LR Image from the original HR Image using the LR Image transformer
    lr_image = lr_scale(hr_image)

    ## Create the restored HR Image from the LR Image using the classical method of restored HR Image transforms
    hr_restore_img = hr_scale(lr_image)

    return to_tensor(lr_image), to_tensor(hr_restore_img), to_tensor(hr_image)

def run_pipeline(val_data_list, batch_size=1):
    """
    Run the pipeline for naive non deep learning super resolution approach (bicubic interpolation)

    Args:
        val_data_list (list): List of paths to the images to be super resolved.
        batch_size (int): Batch size to be used for super resolution.
    
    Returns:
        results (dict): Dictionary containing the results of the non DL super resolution pipeline.
    """

    ## Create a dictionary to store the results
    results = {
                    "mse": 0,
                    "ssims": 0,
                    "psnr": 0,
                    "ssim": 0,
                    "batch_sizes": 0,
                }
    
    ## Create a progress bar
    val_bar = tqdm(val_data_list, total=len(val_data_list))

    ## Iterate over the images
    for image_path in val_bar:

        ## Increment the number of images
        results["batch_sizes"] += batch_size

        ## Get the LR, restored HR and HR images using the naive super resolution method
        lr_img, hr_restore, hr_img = naive_super_resolution(image_path, lr_scale=256, hr_scale=1024)

        ## Calculate the MSE for current image
        batch_mse = ((hr_restore - hr_img) ** 2).data.mean()

        ## Store the MSE for current image
        results["mse"] += batch_mse * batch_size
        
        ## Calculate the SSIM for current image
        batch_ssim = ssim(hr_restore.unsqueeze(0), hr_img).item()

        ## Store the SSIM for current image
        results["ssims"] += batch_ssim * batch_size

        ## Calculate the PSNR for current image
        results["psnr"] = 10 * math.log10((hr_img.max() ** 2)/ (results["mse"] / results["batch_sizes"]))
        
        ## Calculate the SSIM for all processed images
        results["ssim"] = (results["ssims"] / results["batch_sizes"])

        ## Update the progress bar
        val_bar.set_description(desc="PSNR: %.4f dB SSIM: %.4f"% (results["psnr"], results["ssim"]))

    return results

if __name__ == "__main__":

    ## Load the validation data list
    with open('../data/val_images.pkl', 'rb') as f:
        val_data_list = pickle.load(f)
    
    ## Run the pipeline
    results = run_pipeline(val_data_list, batch_size=1)

    ## Print the results
    print("PSNR: %.4f dB SSIM: %.4f"% (results["psnr"], results["ssim"]))

    ## Save the results
    data_frame = pd.DataFrame(data=results, index=range(1, 2))
    data_frame.to_csv("../logs/non_dl_approach_metrics.csv", index_label="Iteration")
