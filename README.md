# Computer Vision Project: Coin Detection and Image Stitching

This repository contains Python scripts for performing two computer vision tasks:
1. Coin detection, segmentation, and counting
2. Image stitching to create panoramas

## Project Structure

```
IMT2022089_VR_ASSIGNMENT1/
├── Assignment1/
│   ├── EdgeDetection_Results_Image1/  # Output from coin detection for Image1
│   ├── EdgeDetection_Results_Image2/  # Output from coin detection for Image2
│   ├── EdgeDetection_Results_Image3/  # Output from coin detection for Image3
│   ├── Images/                        # Contains input images for coin detection
│   │   ├── Image1.jpeg
│   │   ├── Image2.jpeg
│   │   └── Image3.jpeg
│   ├── PanoramaResults_Stitching1/    # Stitching results for folder Stitching1
│   │   ├── composite_0.jpg            # Initial composite (first image)
│   │   ├── composite_1.jpg            # Result after blending second image
│   │   ├── image_1_with_features.jpg  # First image with detected features
│   │   ├── image_2_with_features.jpg  # Second image with detected features
│   │   ├── matches_1to2.jpg           # Visualization of matches between images
│   │   └── stitched_panorama_Stitching1.jpg  # Final panorama result
│   ├── PanoramaResults_Stitching2/    # Stitching results for folder Stitching2
│   │   ├── composite_0.jpg            # Initial composite (first image)
│   │   ├── composite_1.jpg            # Result after blending second image
│   │   ├── composite_2.jpg            # Result after blending third image
│   │   ├── image_1_with_features.jpg  # First image with detected features
│   │   ├── image_2_with_features.jpg  # Second image with detected features
│   │   ├── image_3_with_features.jpg  # Third image with detected features
│   │   ├── matches_1to2.jpg           # Matches between first and second images
│   │   ├── matches_2to3.jpg           # Matches between second and third images
│   │   └── stitched_panorama_Stitching2.jpg  # Final panorama result
│   ├── Processed_Coins_Image1/        # Contains extracted coins from Image1
│   ├── Processed_Coins_Image2/        # Contains extracted coins from Image2
│   ├── Processed_Coins_Image3/        # Contains extracted coins from Image3
│   ├── Stitching1/                    # Contains 2 split images for panorama stitching
│   │   ├── *.jpg
│   └── Stitching2/                    # Contains 3 split images for panorama stitching
│       ├── *.jpg
├── coins.py                           # Script for coin detection and analysis
└── Stitching.py                       # Script for panorama creation
```

## Coin Detection and Analysis (coins.py)

This script processes images containing coins and performs three main tasks:

### a. Coin Detection 
- Uses multiple edge detection techniques (Canny, Sobel, Laplacian)
- Visualizes detected coins by outlining them in the original image
- Outputs edge maps to the EdgeDetection_Results folders

### b. Coin Segmentation
- Extracts each individual coin from the image
- Uses adaptive thresholding and contour detection
- Outputs individual coin images to the Processed_Coins folders
- Creates a visualization overlay (detected_coins_overlay.png) in the Processed_Coins folder showing all detected coins

### c. Coin Counting
- Counts the total number of coins in each image
- Filters out non-coin objects using size and circularity criteria
- Prints the count to the console

### Methods Implemented
- Image enhancement with Gaussian blur
- Adaptive thresholding for binary mask creation
- Morphological operations for mask refinement
- Contour detection to identify coin boundaries
- Circularity calculation to filter out non-coin objects

## Image Stitching (Stitching.py)

This script creates panoramic images by stitching together multiple image segments.

## Features

- Processes two sets of images (Stitching1 with 2 images, Stitching2 with 3 images)
- SIFT feature detection for robust key point extraction
- Homography-based image alignment
- Distance-weighted blending for seamless transitions
- Automatic dark edge trimming for clean results
- Visualization of detected features and matches
- Support for processing multiple folders of images
- Results will be saved in folders named "PanoramaResults_Stitching1" and "PanoramaResults_Stitching2"

## Output Files

For each processed folder, the following files are generated:

- `image_X_with_features.jpg`: Shows detected features for each input image
- `matches_XtoY.jpg`: Visualizes the matching features between pairs of images
- `composite_X.jpg`: Intermediate results after blending X+1 images
- `stitched_panorama_[foldername].jpg`: The final panorama image

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Standard libraries: os, glob, shutil

## Installation

```bash
pip install opencv-python numpy
```

## How to Run

### Clone the repository
```
https://github.com/Aayush-Bhargav/VR_Assignment1_AayushBhargav_IMT2022089.git
```
### Navigate to the right directory
```
cd IMT2022089_VR_Assignment1
```
### Coin Detection
```bash
python3 coins.py
```
This will process all three images in the Images folder and output the results to the respective folders.

### Image Stitching
```bash
python3 Stitching.py
```
This will process the image sets in Stitching1 and Stitching2 folders and create panoramic images.

## Output

### Coin Detection
- Edge maps using different detection methods
- Visualization of detected coin boundaries
- Individual extracted coin images
- detected_coins_overlay.png showing all identified coins
- Console output with the count of coins detected

### Image Stitching
- Visualizations of detected features in each input image
- Visualizations of matched features between consecutive images
- Final stitched panorama images
- Step-by-step processing visualizations

## Notes
- The coin detection algorithm is designed to handle various lighting conditions and coin arrangements
- All input images are standardized to a consistent width before processing
