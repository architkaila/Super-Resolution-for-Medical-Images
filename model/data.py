import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

def is_image_file(filename):
    """
    Checks if a file is in any of the correct image format.

    Args:
        filename (string): Path to the file to be checked.
    
    Returns:
        valid_img (bool): True if the file is in any of the correct image format, False otherwise.
    """

    ## Mark image valid if it is any of the allowed formats
    valid_img = any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    
    return valid_img


def calculate_valid_crop_size(crop_size, upscale_factor):
    """
    Calculates the valid crop size for the given crop size and upscale factor.

    Args:
        crop_size (int): Size of the crop to be taken from the image.
        upscale_factor (int): The factor by which the image will be upsampled.
    
    Returns:
        crop_size (int): Valid crop size for the given crop size and upscale factor.
    """
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    """
    Creates the transforms for the High Resolution Images. Includes Random Crop, 
    Random Vertical Flip, Random Horizontal Flip and To Tensor.

    Args:
        crop_size (int): Size of the crop to be taken from the image.
    
    Returns:
        transforms (Compose): Composed transforms for the High Resolution Images.
    """
    return transforms.Compose([
        #transforms.RandomCrop(crop_size, pad_if_needed=True),
        #transforms.RandomVerticalFlip(p=0.3),
        #transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    """
    Creates the transforms for the Low Resolution Images. Includes Resize, To Tensor.

    Args:
        crop_size (int): Size of the crop to be taken from the image.
        upscale_factor (int): The factor by which the image will be upsampled.
    
    Returns:
        transforms (Compose): Composed transforms for the Low Resolution Images.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

## Transforms for displaying the images
display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    #transforms.CenterCrop(400),
    transforms.ToTensor()
])


class TrainDataset(Dataset):
    """
    Class to load the training dataset. This class reads the images from the dataset directory 
    and applies the required transformations.
    """
    
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        """
        The constructor for TrainDataset class.

        Args:
            dataset_dir (string): Path to the train dataset directory on the disk.
            crop_size (int): Size of the crop to be taken from the image.
            upscale_factor (int): The factor by which the image will be upsampled.
        """
        super(TrainDataset, self).__init__()

        ## Initialize the required variables

        # List of image filenames
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        
        # Calculate the valid crop size
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        # Create the transforms for the High Resolution Images
        self.hr_transform = train_hr_transform(crop_size)

        # Create the transforms for the Low Resolution Images
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        """
        Function to get the image at the given index.

        Args:
            index (int): Index of the image to be returned.
        
        Returns:
            lr_image (Tensor): Low Resolution Image
            hr_image (Tensor): High Resolution Image
        """

        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        """
        Function to get the length of the train dataset.
        """
        return len(self.image_filenames)


class ValDataset(Dataset):
    
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        """
        The constructor for ValDataset class.

        Args:
            dataset_dir (string): Path to the val dataset directory on the disk.
            crop_size (int): Size of the crop to be taken from the image.
            upscale_factor (int): The factor by which the image will be upsampled.
        """
        super(ValDataset, self).__init__()

        ## Initialize the required variables

        # List of image filenames
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

        # Calculate the valid crop size
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        # Store the upscale factor
        self.upscale_factor = upscale_factor


    def __getitem__(self, index):
        """
        Function to get the image at the given index.

        Args:
            index (int): Index of the image to be returned.
        
        Returns:
            lr_image (Tensor): Low Resolution Image
            hr_restore_img (Tensor): High Resolution Image after classical restoration process
            hr_image (Tensor): High Resolution Image
        """

        ## Read the actual HR image
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        
        ## Create the LR image transformer by downsampling the HR image and applying bicubic interpolation
        lr_scale = transforms.Resize((256,256), interpolation=Image.BICUBIC)

        ## Create the restored HR image tranformer (simple classical method) by upsampling the LR image and applying bicubic interpolation
        hr_scale = transforms.Resize((1024,1024), interpolation=Image.BICUBIC)
        
        ## Create the HR Image cropped to the valid crop size
        #hr_image = transforms.CenterCrop(self.crop_size)(hr_image)

        ## Create the LR Image from the original HR Image using the LR Image transformer
        lr_image = lr_scale(hr_image)

        ## Create the restored HR Image from the LR Image using the classical method of restored HR Image transforms
        hr_restore_img = hr_scale(lr_image)

        return to_tensor(lr_image), to_tensor(hr_restore_img), to_tensor(hr_image)

    def __len__(self):
        """
        Function to get the length of the val dataset.
        """
        return len(self.image_filenames)