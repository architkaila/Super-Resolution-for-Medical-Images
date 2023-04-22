import os
import math
import pandas as pd
import torch
import torchvision
from prepare_data import TrainDataset, ValDataset
from torch.utils.data import DataLoader
from model_architecture import Generator, Discriminator
from custom_loss import GeneratorLoss
from model_metrics import ssim
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
import pickle

## Set the seed for reproducibility
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)


def run_pipeline(arguments):
    """
    This function runs the training pipeline.
    
    Args:
        arguments: Argparse object containing the arguments passed to the script

    Returns:
        None
    """

    ## Create directories to store results if they don't exist
    print("[INFO] Creating directories")
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    ## Initialize parameters from arguments
    UPSCALE_FACTOR = arguments.upscale_factor
    NUM_EPOCHS = arguments.num_epochs
    BATCH_SIZE = arguments.batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Load Train and Val data splits
    with open('./data/train_images.pkl', 'rb') as f:
        train_data_list = pickle.load(f)
    with open('./data/val_images.pkl', 'rb') as f:
        val_data_list = pickle.load(f)
    
    ## Load the train dataset
    print("[INFO] Loading Train dataset")
    train_set = TrainDataset(train_data_list)

    ## Load the validation dataset
    print("[INFO] Loading Val dataset")
    val_set = ValDataset(val_data_list)

    ## Create the train data loader
    print("[INFO] Creating Train data loader")
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    ## Create the validation data loader
    print("[INFO] Creating Val data loader")
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    ## Initialize the Generator model
    netG = Generator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))

    ## Initialize the Discriminator model
    netD = Discriminator().to(DEVICE)
    print("# discriminator parameters:", sum(param.numel() for param in netD.parameters()))

    ## Initialize the loss function
    generator_criterion = GeneratorLoss().to(DEVICE)

    ## Initialize the optimizer
    optimizerG = torch.optim.AdamW(netG.parameters(), lr=1e-3)
    optimizerD = torch.optim.AdamW(netD.parameters(), lr=1e-3)

    ## Initialize the dictionary to store the results
    results = {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }

    ## Start training
    for epoch in range(1, NUM_EPOCHS + 1):

        ## Progress bar for training
        train_bar = tqdm(train_loader, total=len(train_loader))

        ## Initialize the dictionary to store the results
        running_results = {"batch_sizes": 0, 
                           "d_loss": 0, "g_loss": 0,
                           "d_score": 0, "g_score": 0,
                        }

        ## Set the models to train mode
        netG.train()
        netD.train()

        ## Iterate over the batch of images
        for lr_img, hr_img in train_bar:
            
            ## Get the current batch size
            batch_size = lr_img.size(0)
            running_results["batch_sizes"] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################

            ## Move the images to the GPU
            hr_img = hr_img.to(DEVICE) # high resolution image
            lr_img = lr_img.to(DEVICE) # low resolution image
            with torch.no_grad():
                sr_img = netG(lr_img) # super resolution image
            
            ## Set the gradients of Discriminator to zero
            netD.zero_grad()
            
            ## Formward propagate the HR image and SR image through the discriminator
            real_out = netD(hr_img).mean()
            fake_out = netD(sr_img).mean()
            
            ## Calculate the discriminator loss
            d_loss = 1 - real_out + fake_out

            ## Backpropagate the loss
            d_loss.backward(retain_graph=True)

            ## Update the weights
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            
            ## Set the gradients of the Generator to zero
            netG.zero_grad()

            ## Forward propagate the LR image through the generator to get the SR image
            sr_img = netG(lr_img)

            ## Forward propagate the SR image through the discriminator
            with torch.no_grad():
                fake_out = netD(sr_img).mean()
            
            ## Calculate the generator loss
            g_loss = generator_criterion(fake_out, sr_img, hr_img)

            ## Backpropagate the loss
            g_loss.backward()

            ## Update the weights
            optimizerG.step()

            ## Store loss for current batch
            running_results["g_loss"] += g_loss.item() * batch_size
            running_results["d_loss"] += d_loss.item() * batch_size
            running_results["d_score"] += real_out.item() * batch_size
            running_results["g_score"] += fake_out.item() * batch_size

            ## Update the progress bar and print the results
            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    NUM_EPOCHS,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        ## Set the Generator to evaluation mode
        netG.eval()

        ## Run the validation loop
        with torch.no_grad():

            ## Progress bar for validation loop
            val_bar = tqdm(val_loader, total=len(val_loader))
            
            ## Initialize the dictionary to store the results
            valing_results = {
                "mse": 0,
                "ssims": 0,
                "psnr": 0,
                "ssim": 0,
                "batch_sizes": 0,
            }
            
            ## Iterate over the batch of images
            for val_lr, val_hr in val_bar:
                
                ## Get the current batch size
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size

                ## Move the images to the GPU
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                
                ## Forward propagate the LR image through the generator to get the SR image
                sr = netG(lr)

                ## Calculate All the metrics
                ## Calculate and store the MSE
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size

                ## Calculate and store the SSIMs
                batch_ssim = ssim(sr, hr).item()
                valing_results["ssims"] += batch_ssim * batch_size
                
                ## Calculate and store the PSNR
                valing_results["psnr"] = 10 * math.log10( (hr.max() ** 2) / (valing_results["mse"] / valing_results["batch_sizes"]))
                
                ## Calculate and store the SSIM
                valing_results["ssim"] = (valing_results["ssims"] / valing_results["batch_sizes"])

                ## Update the progress bar and print the results
                val_bar.set_description(
                    desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                    % (valing_results["psnr"], valing_results["ssim"])
                )

        ## Set the Generator and Discriminator to train mode
        netG.train()
        netD.train()

        ## Save the Generator model
        torch.save({"model": netG.state_dict()},
            f"./models/netG_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar")
        
        ## Save the Discriminator model
        torch.save({"model": netD.state_dict()},
            f"./models/netD_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar")

        ## Store the losses and scores for the current epoch
        results["d_loss"].append(running_results["d_loss"] / running_results["batch_sizes"])
        results["g_loss"].append(running_results["g_loss"] / running_results["batch_sizes"])
        results["d_score"].append(running_results["d_score"] / running_results["batch_sizes"])
        results["g_score"].append(running_results["g_score"] / running_results["batch_sizes"])
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])
        
        ## Save the results every epoch
        out_path = "./logs/"
        data_frame = pd.DataFrame(
            data={
                "Loss_D": results["d_loss"],
                "Loss_G": results["g_loss"],
                "Score_D": results["d_score"],
                "Score_G": results["g_score"],
                "PSNR": results["psnr"],
                "SSIM": results["ssim"],
            },
            index=range(1, epoch + 1),
        )

        ## Save the results to the disk
        data_frame.to_csv(
            out_path + "metrics_" + str(epoch) + "_train_results.csv",
            index_label="Epoch",
        )


if __name__ == '__main__':

    ## Initialize argument parser
    parser = argparse.ArgumentParser()

    ## Add arguments
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='number of item in a batch')

    ## Parse arguments
    arguments = parser.parse_args()

    ## Train model
    run_pipeline(arguments)