import pandas as pd
import random
import os

def create_train_test_csv(input_csv, train_csv, val_csv, val_ratio=0.1):
    # Read the input CSV file
    data = pd.read_csv(input_csv)

    # Get the list of image filenames
    image_filenames = data['crop_filepath'].tolist()

    # Shuffle the filenames randomly
    random.shuffle(image_filenames)

    # Calculate the number of images for the test set
    num_test = int(len(image_filenames) * val_ratio)

    # Split the filenames into train and test sets
    train_filenames = image_filenames[num_test:]
    val_filenames = image_filenames[:num_test]

    # Create DataFrames for train and test sets
    train_df = pd.DataFrame({'Image Filename': train_filenames})
    val_df = pd.DataFrame({'Image Filename': val_filenames})

    # Write train and test DataFrames to CSV files
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

# Example usage:
input_csv = '/Volumes/Finkbeiner-Steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/Parker-SemiSupervised/XENum_image_data_train_test_plaque_label_APOE_filtered.csv'  # Change this to your input CSV file path
train_csv = '/Volumes/Finkbeiner-Steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/Parker-SemiSupervised/train.csv'       # Change this to the desired train CSV file path
val_csv = '/Volumes/Finkbeiner-Steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/Parker-SemiSupervised/val.csv'         # Change this to the desired test CSV file path
create_train_test_csv(input_csv, train_csv, val_csv, val_ratio=0.1)
