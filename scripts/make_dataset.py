import opendatasets as od

def download_dataset(dataset_url):
    """
    This function downloads the dataset from Kaggle

    Args:
        dataset_url (str): The URL of the dataset

    Returns:
        None
    """
    od.download(dataset_url)

if __name__ == "__main__":

    ## Download the dataset
    dataset_url = 'https://www.kaggle.com/andrewmvd/heart-failure-clinical-data'
    download_dataset(dataset_url)