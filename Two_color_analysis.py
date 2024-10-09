# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:18:25 2024

@author: el4003ak
"""

import os
import cv2
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt

folder1 = r'E:\Elham\Two color experiment 05102024\BSA_CS_6h\20X_FS_blue_mix_CS_BSA_6h_scan_large'  # Images assigned to the Red channel
folder2 = r'E:\Elham\Two color experiment 05102024\BSA_CS_6h\20X_FS_red_mix_CS_BSA_6h_scan_large'  # Images assigned to the Green channel
output_folder = r'E:\Elham\Two color experiment 05102024\BSA_CS_6h\merge'


#Combination of the two codes of blurness and merging. The code check the blurness of the image and if it is less than 1.5, then it merges the two colors


# Function to calculate blurriness
def calculate_blurriness(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the Laplacian of the image and then the variance
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance_of_laplacian = laplacian.var()
    
    # Print the blurriness value
    #print(f"Blurriness (Variance of Laplacian): {variance_of_laplacian}")
    #plt.hist(variance_of_laplacian)
    
    
    # Display the original image and grayscale using Matplotlib
    plt.figure(figsize=(10, 5))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for displaying correctly
    plt.title("Original Image")
    plt.axis("off")
    
    # Plot the grayscale image
    plt.subplot(1, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")
    
    # Show the plots
    plt.show()
    
    return variance_of_laplacian

image=cv2.imread(r'E:\Elham\Two color experiment 05102024\BSA_S_30min\20X_FS_blue_mix_S_BSA_30min_scan_large\tile_x004_y007.tif')

calculate_blurriness(image)

# Function to merge two images
def merge_images(image1_path, image2_path, output_path):
    # Read the images using cv2
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        print(f"Skipping {image1_path} and {image2_path} due to size mismatch.")
        return
    
    # Create an empty color image with three channels (Red, Green, Blue)
    merged_image = cv2.merge([img1 * 0, img2, img1])  # Green for img2, Red for img1, Blue remains 0
    
    # Save the merged image
    cv2.imwrite(output_path, merged_image)
    print(f"Merged and saved: {output_path}")


# Get the list of files in both folders
folder1_images = {Path(f).stem: f for f in os.listdir(folder1) if f.endswith('.tif')}
folder2_images = {Path(f).stem: f for f in os.listdir(folder2) if f.endswith('.tif')}

# Process images with the same name in both folders
for image_name in folder1_images.keys() & folder2_images.keys():
    image1_path = os.path.join(folder1, folder1_images[image_name])
    image2_path = os.path.join(folder2, folder2_images[image_name])
    
    # Read images to calculate blurriness
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Calculate blurriness for both images
    blurriness_img1 = calculate_blurriness(img1)
    blurriness_img2 = calculate_blurriness(img2)

    # Check if either image is too blurry
    if blurriness_img1 > 3 or blurriness_img2 > 3:
        print(f"Skipping merging for {image_name} due to blurriness.")
        continue  # Skip merging this pair

    # Output file path
    output_path = os.path.join(output_folder, f"{image_name}_merged.tif")
    
    # Merge the images and save
    merge_images(image1_path, image2_path, output_path)
    




    
#2nd step would be Object detection and thresholding

# Paths
input_folder = r'E:\Elham\Two color experiment 05102024\BSA_CS_6h\merge'  # Merged images with Red and Green channels
output_folder = r'E:\Elham\Two color experiment 05102024\BSA_CS_6h\masks'
csv_output = r'E:\Elham\Two color experiment 05102024\BSA_CS_6h\masks\object_report.csv'


    
# Threshold values for Red and Green channels
red_threshold = (6, 255)  # Modify according to your needs
green_threshold = (6, 255)  # Modify according to your needs

# Function to threshold images and detect contours
def threshold_and_detect(image, color_channel, threshold_value):
    # Threshold the image based on the given channel and threshold range
    _, thresholded = cv2.threshold(image[:, :, color_channel], threshold_value[0], threshold_value[1], cv2.THRESH_BINARY)
    
    # Find contours (objects)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return thresholded, contours

# Function to find overlapping objects
def find_overlapping_objects(red_mask, green_mask):
    # Bitwise AND to find overlapping areas in both masks
    overlap_mask = cv2.bitwise_and(red_mask, green_mask)
    
    # Find contours in the overlapping mask
    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return overlap_mask, contours

# Open CSV file to write the results
with open(csv_output, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Red Objects', 'Green Objects', 'Overlapping Objects'])  # CSV headers

    # Process all merged images in the input folder
    for merged_image_name in os.listdir(input_folder):
        if merged_image_name.endswith(".tif"):
            merged_image_path = os.path.join(input_folder, merged_image_name)
            merged_image = cv2.imread(merged_image_path)

            # Thresholding for Red and Green channels
            red_mask, red_contours = threshold_and_detect(merged_image, color_channel=2, threshold_value=red_threshold)  # Red channel (2)
            green_mask, green_contours = threshold_and_detect(merged_image, color_channel=1, threshold_value=green_threshold)  # Green channel (1)

            # Save the threshold masks
            red_mask_path = os.path.join(output_folder, f"{Path(merged_image_name).stem}_red_mask.tif")
            green_mask_path = os.path.join(output_folder, f"{Path(merged_image_name).stem}_green_mask.tif")
            cv2.imwrite(red_mask_path, red_mask)
            cv2.imwrite(green_mask_path, green_mask)

            # Detect overlapping objects between Red and Green
            overlap_mask, overlap_contours = find_overlapping_objects(red_mask, green_mask)

            # Save the overlap mask
            overlap_mask_path = os.path.join(output_folder, f"{Path(merged_image_name).stem}_overlap_mask.tif")
            cv2.imwrite(overlap_mask_path, overlap_mask)

            # Save the final output image with drawn contours (Optional)
            output_image = merged_image.copy()
            cv2.drawContours(output_image, red_contours, -1, (0, 0, 255), 2)  # Red contours
            cv2.drawContours(output_image, green_contours, -1, (0, 255, 0), 2)  # Green contours
            cv2.drawContours(output_image, overlap_contours, -1, (255, 255, 0), 2)  # Overlapping contours
            final_output_path = os.path.join(output_folder, f"{Path(merged_image_name).stem}_contours.tif")
            cv2.imwrite(final_output_path, output_image)

            # Write the detection results to the CSV file
            writer.writerow([
                merged_image_name,                # Image name
                len(red_contours),                # Number of red objects
                len(green_contours),              # Number of green objects
                len(overlap_contours)             # Number of overlapping objects
            ])

            # Print the results for reference
            print(f"Image: {merged_image_name}")
            print(f"Red objects detected: {len(red_contours)}")
            print(f"Green objects detected: {len(green_contours)}")
            print(f"Overlapping objects detected: {len(overlap_contours)}")
            
            
            
#Measuring the blurness of the image with methods other than Laplacian including local laplacian and FFT

# Using local focus detection:
def check_focus_in_regions(image, region_size, threshold):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the image dimensions
    height, width = gray_image.shape
    
    # Iterate through the image by regions
    for y_pos in range(0, height, region_size[1]):
        for x_pos in range(0, width, region_size[0]):
            # Extract the region of interest (ROI)
            roi = gray_image[y_pos:y_pos + region_size[1], x_pos:x_pos + region_size[0]]
            
            # Compute the Laplacian of the region and calculate its variance
            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            variance_of_laplacian = laplacian.var()
            
            # Determine focus status for the region
            focus_status = "In Focus" if variance_of_laplacian >= threshold else "Out of Focus"
            
            # Draw a rectangle and label the region on the original image
            color = (0, 255, 0) if focus_status == "In Focus" else (0, 0, 255)
            cv2.rectangle(image, (x_pos, y_pos), (x_pos + region_size[0], y_pos + region_size[1]), color, 2)
            cv2.putText(image, f"{focus_status}", (x_pos, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the original image with focus status in regions
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Focus Detection in Regions")
    plt.axis("off")
    plt.show()



# Check the focus in regions of the image
check_focus_in_regions(image, region_size=(30, 30), threshold=100.0)

#In this code Rectangles are drawn around each region to indicate whether it is in focus (green) or out of focus (red).


#Using FFT
#Sharp (in-focus) images contain more high-frequency components, while blurred (out-of-focus) images contain mostly low-frequency components.
def fft_blurriness(image, threshold=0.5):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Fast Fourier Transform (FFT)
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    
    # Analyze high-frequency components
    high_freq = np.sum(magnitude_spectrum > threshold)
    
    # Display the magnitude spectrum
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum (FFT)")
    plt.axis("off")
    
    plt.show()

    return high_freq


# Analyze the frequency components
high_freq = fft_blurriness(image, threshold=2)
print(f"High frequency components: {high_freq}")



#Plotting the data

import pandas as pd

# Function to calculate the ratio for each CSV file
def calculate_ratio(folder_path):
    ratios = {}
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Calculate new columns for "Red Objects - Overlapping Objects" and "Green Objects - Overlapping Objects"
            red_minus_overlap = df["Red Objects"] - df["Overlapping Objects"]
            green_minus_overlap = df["Green Objects"] - df["Overlapping Objects"]
            
            # Calculate sums
            sum_red_minus_overlap = red_minus_overlap.sum()
            sum_green_minus_overlap = green_minus_overlap.sum()
            sum_overlap = df["Overlapping Objects"].sum()
            
            # Calculate the ratio: sum of overlap divided by (sum of red - overlap + sum of green - overlap)
            total_new_sum = sum_red_minus_overlap + sum_green_minus_overlap
            if total_new_sum > 0:  # To avoid division by zero
                ratio = sum_overlap / total_new_sum
            else:
                ratio = 0  # If the total is zero, set ratio to 0
            
            # Store the ratio with the file name (without extension) as key
            ratios[os.path.splitext(file_name)[0]] = ratio
    
    return ratios


# Function to plot the bar chart
def plot_ratios(ratios, output_path):
    # File names and corresponding ratios
    file_names = list(ratios.keys())
    ratio_values = list(ratios.values())
    
    # Plotting the bar chart
    plt.figure(figsize=(30, 18))
    
    # Create the bar chart with orange color and opacity
    plt.bar(file_names, ratio_values, color='orange', alpha=0.9, width=0.5)  # Narrower bars with width=0.5
    
    # Set y-axis limits and font sizes
    plt.ylim(0, 1)
    plt.yticks(fontsize=30)
    
    # Set y-axis label
    plt.ylabel("Ratio of the two-color clusters to one-color clusters", fontsize=30)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=30)
    
    # Save the plot as a PDF file
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    # plt.close()

# Example usage
folder_path = r'E:\Elham\Two color experiment 05102024\Excel_files\BSA_stationary_Overtime'  # Replace with your folder path
output_plot_path = r'E:\Elham\Two color experiment 05102024\Excel_files\BSA_stationary_Overtime\plot.pdf'  # Replace with your desired output path
ratios = calculate_ratio(folder_path)
plot_ratios(ratios, output_plot_path)

