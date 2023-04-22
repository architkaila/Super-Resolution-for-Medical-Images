import random
import os
import glob
import pickle


def split_data():
    """
    This function splits the image data into train and test

    Args:
        None

    Returns:
        None
    """
    ## Get list of all images in full dataset
    all_images_list = glob.glob(f"./dataset/*/*/*.png", recursive=False)

    ## Shuffle the data
    random.shuffle(all_images_list)

    ## Split the data into train and test
    train_images = all_images_list[:90000]
    test_images = all_images_list[90000:]

    ## Save the train and test split
    with open("./dataset/train_images.pkl", "wb") as fp:
        pickle.dump(train_images, fp)
    with open("./dataset/val_images.pkl", "wb") as fp:
        pickle.dump(test_images, fp)


if __name__ == "__main__":

    ## Create the train and test data
    split_data()