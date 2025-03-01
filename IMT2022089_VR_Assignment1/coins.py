# Import necessary libraries
import cv2
import numpy as np
import os
import shutil

def extract_and_count_coins(source_image_path, destination_folder):
    """
    Segments and extracts individual coins from an image, saving them to a specified directory. Also, counts the coins and displays the total coins in the image.
    """
    # Read the source image
    original_img = cv2.imread(source_image_path)
    
    if original_img is None:
        print(f"Error: Unable to load source image: {source_image_path}")
        return
    
    # Setup output directories for extracted coins
    coin_storage_path = os.path.join(destination_folder, "ExtractedCoins")

    # Remove existing directory if it exists to start fresh
    if os.path.exists(coin_storage_path):
        shutil.rmtree(coin_storage_path)

    # Create fresh directory
    os.makedirs(coin_storage_path, exist_ok=True)
    
    # Convert to grayscale for processing
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    smooth_grayscale = cv2.GaussianBlur(grayscale_img, (9, 9), 1.5)

    # Use adaptive thresholding to identify coin regions
    binary_mask = cv2.adaptiveThreshold(smooth_grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Clean up the binary mask with morphological operations
    structuring_element = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, structuring_element, iterations=2)

    # Find coin boundaries in the refined mask
    boundaries, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image for visualization
    visualization_canvas = np.zeros_like(original_img)

    coin_counter = 1

    # Process each potential coin region
    for idx, boundary in enumerate(boundaries):
        region_size = cv2.contourArea(boundary)
        
        # Filter out regions that are too small or too large to be coins
        if region_size < 5000 or region_size > 35000:  
            continue

        # Calculate shape metrics to verify circular objects
        boundary_length = cv2.arcLength(boundary, True)
        roundness_factor = 4 * np.pi * (region_size / (boundary_length ** 2))
        
        # Only keep objects that are sufficiently circular
        if roundness_factor < 0.7:  
            continue

        # Find the best-fit circle for the coin
        (center_x, center_y), circle_radius = cv2.minEnclosingCircle(boundary)
        circle_center = (int(center_x), int(center_y))
        circle_radius = int(circle_radius)

        # Create a circular mask for precise extraction
        coin_mask = np.zeros_like(grayscale_img)
        cv2.circle(coin_mask, circle_center, circle_radius, 255, thickness=cv2.FILLED)

        # Apply the mask to extract just the coin
        isolated_coin = cv2.bitwise_and(original_img, original_img, mask=coin_mask)

        # Crop to the coin's bounding box
        x_min, y_min = circle_center[0] - circle_radius, circle_center[1] - circle_radius
        width, height = 2 * circle_radius, 2 * circle_radius
        coin_extract = isolated_coin[y_min:y_min+height, x_min:x_min+width]

        # Save the extracted coin image
        output_path = os.path.join(coin_storage_path, f"extracted_coin_{coin_counter}.png")
        cv2.imwrite(output_path, coin_extract)
        coin_counter += 1

        # Mark the detected coin on visualization image
        cv2.circle(visualization_canvas, circle_center, circle_radius, (0, 255, 0), thickness=-1)

    # Save the visualization showing detected coins
    visualization_path = os.path.join(destination_folder, "detected_coins_overlay.png")
    cv2.imwrite(visualization_path, visualization_canvas)
    print(f"Extracted coins saved to '{coin_storage_path}'")
    print(f"Visualization image saved to '{visualization_path}'")
    print(f"Total coin count: {coin_counter-1}")


def enhance_image(input_path):
    """
    Prepares an image for analysis by converting to grayscale and reducing noise.
    """
    source_img = cv2.imread(input_path)
    grayscale = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grayscale, (9,9), 1.456)
    return source_img, grayscale, denoised


def create_edge_map_canny(denoised_img):
    """
    Applies Canny edge detection algorithm.
    """
    return cv2.Canny(denoised_img, 50, 150)


def create_edge_map_laplacian(grayscale_img):
    """
    Applies Laplacian edge detection.
    """
    smoothed = cv2.GaussianBlur(grayscale_img, (13, 13), 2)
    laplacian_result = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=5)
    return cv2.convertScaleAbs(laplacian_result)


def create_edge_map_sobel(grayscale_img):
    """
    Applies Sobel edge detection by combining x and y gradients.
    """
    # Apply additional smoothing for better gradient detection
    smoothed = cv2.GaussianBlur(grayscale_img, (17, 17), 2.5)
    
    # Calculate x and y gradients
    gradient_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    
    # Combine gradients
    combined_gradient = cv2.convertScaleAbs(gradient_x) + cv2.convertScaleAbs(gradient_y)
    
    # Apply threshold to create binary edge map
    _, binary_edges = cv2.threshold(combined_gradient, 120, 255, cv2.THRESH_BINARY)
    return binary_edges


def identify_and_outline_objects(edge_map, original_img):
    """
    Identifies object boundaries from an edge map and highlights them on the original image.
    """
    object_boundaries, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = original_img.copy()
    cv2.drawContours(result_img, object_boundaries, -1, (0, 255, 0), 2)
    return result_img, object_boundaries


def export_result(img_data, output_dir, filename):
    """
    Saves processed image to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{filename}.jpg"), img_data)

def process_image(image_filename):
    """
    Process a single image with appropriate folder naming
    """
    # Extract image name without extension
    image_name = os.path.splitext(os.path.basename(image_filename))[0]
    
    # Create image-specific output folders
    output_dir = f"EdgeDetection_Results_{image_name}"
    processed_dir = f"Processed_Coins_{image_name}"
    
    # Full path to the image
    image_path = os.path.join("Images", image_filename)
    
    print(f"\nProcessing {image_filename}...")
    
    # Step 1: Enhance the image
    original, grayscale, denoised = enhance_image(image_path)

    # Step 2: Apply different edge detection methods
    canny_result = create_edge_map_canny(denoised)
    sobel_result = create_edge_map_sobel(grayscale)
    laplacian_result = create_edge_map_laplacian(grayscale)

    # Step 3: Identify and outline coins using Canny edges
    outlined_image, boundaries = identify_and_outline_objects(canny_result, original)

    # Step 4: Save the edge detection results
    export_result(canny_result, output_dir, "Canny_EdgeMap")
    export_result(sobel_result, output_dir, "Sobel_EdgeMap")
    export_result(laplacian_result, output_dir, "Laplacian_EdgeMap")
    export_result(outlined_image, output_dir, "Coin_Boundaries")

    # Step 5: Process and extract individual coins
    extract_and_count_coins(image_path, processed_dir)

def main():
    """
    Main function to execute the coin detection and analysis pipeline for multiple images.
    """
    # List of images to process
    image_files = ["Image1.jpeg", "Image2.jpeg", "Image3.jpeg"]
    
    # Process each image
    for image_file in image_files:
        process_image(image_file)

    # Clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Entry point for the program
if __name__ == "__main__":
    main()