## Library Imports
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
import argparse

## Local Imports
from model.models import Generator
from model.model_config import DEVICE
from model.model_config import upscale_factor
from model.model_config import model_load_path
from model.model_config import default_image

def prepare_image(image_path, is_hr_image=True):
    """
    Prepare the input image for inference. If the input image is HR, then it is downsampled to 
    LR image. If the input image is LR, then it is made sure the size is 256x256.

    Args:
        image_path (str): Path to the input image.
        is_hr_image (bool): Flag to indicate if the input image is HR or LR.

    Returns:
        input_image (PIL.Image): Input image for inference.
    """
    
    ## Check if the input image is HR or LR
    if is_hr_image:
        ## Load the HR image
        original_hr_image = Image.open(image_path).convert('RGB')

        if original_hr_image.size[0] != 1024 or original_hr_image.size[1] != 1024:
            print("[INFO] Resizing the original input HR image to 1024x1024")
            original_hr_image_scalar = transforms.Resize((1024,1024))
            original_hr_image = original_hr_image_scalar(original_hr_image)

        ## Create the LR image transformer by downsampling the HR image and applying bicubic interpolation
        low_res_image_transform = transforms.Resize((256,256), interpolation=Image.BICUBIC)

        ## Transform HR image to LR Image
        input_image = low_res_image_transform(original_hr_image)
    else:
        ## Load the LR image
        original_lr_image = Image.open(image_path).convert('RGB')

        if original_lr_image.size[0] != 256 or original_lr_image.size[1] != 256:
            print("[INFO] Resizing the original input LR image to 256x256")
            original_lr_image_scalar = transforms.Resize((256,256))
            original_lr_image = original_lr_image_scalar(original_lr_image)

        input_image = original_lr_image

    return input_image

def init_model(model_load_path=model_load_path, upscale_factor=upscale_factor, device=DEVICE):
    """
    Initialize the Generator model and load the weights of the pretrained model.

    Args:
        model_load_path (str): Path to the pretrained model weights.
        upscale_factor (int): Upscale factor for the Generator model.
        device (str): Device to run the model on.

    Returns:
        model (torch.nn.Module): Generator model.
    """
    ## Initialize the Generator model
    model = Generator(upscale_factor=upscale_factor).to(device)

    ## Load the model weights
    state_dict = torch.load(model_load_path, map_location=torch.device(device))
    model.load_state_dict(state_dict["model"])

    ## Set the model to evaluation mode
    model.eval()

    return model

def run_model_inference(low_res_image, model, device="cpu"):
    """
    Run the inference on the input image.

    Args:
        low_res_image (PIL.Image): Input image for inference.
        model (torch.nn.Module): Generator model.
        device (str): Device to run the model on.

    Returns:
        output_sr_image (PIL.Image): Output image from the model.
    """

    ## Move the image to GPU if available
    if device == "cuda":
        low_res_image = low_res_image.cuda()
        
    ## Transform the image to tensor
    low_res_image = to_tensor(low_res_image)
    ## Add a batch dimension
    low_res_image = low_res_image.unsqueeze(0)

    ## Perform model inference
    with torch.no_grad():
        output_sr_image = model(low_res_image)

    ## Remove the batch dimension
    output_sr_image = output_sr_image.squeeze(0)

    ## Transform the tensor to PIL image
    display_transform = transforms.Compose([transforms.ToPILImage()])
    output_sr_image = display_transform(output_sr_image)

    ## Save the output image
    output_sr_image.save("./data/results/output.png")

    print("[INFO] Inference complete, output image saved")

    return output_sr_image


if __name__ == "__main__":
     ## Initialize argument parser
    parser = argparse.ArgumentParser()

    ## Add arguments
    parser.add_argument('--image', default="./data/input.png", type=str, help='Path to the input image')

    ## Parse arguments
    arguments = parser.parse_args()
    
    ## Initialize the model
    print("[INFO] Initializing the model...")
    model = init_model(model_load_path, upscale_factor=upscale_factor, device=DEVICE)

    ## Prepare the input image
    print("[INFO] Preparing the input image...")
    input_image = prepare_image(arguments.image, is_hr_image=True)

    ## Run the inference
    print("[INFO] Running the inference...")
    run_model_inference(input_image, model, device=DEVICE)