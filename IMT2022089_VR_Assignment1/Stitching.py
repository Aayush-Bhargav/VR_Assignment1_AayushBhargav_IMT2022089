
import cv2
import numpy as np
import glob
import os
import shutil

def trim_dark_edges(image): 
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    outlines, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if outlines:
        largest_outline = max(outlines, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_outline)
        return image[y:y+h, x:x+w]
    return image

def visualize_features(img, features):
    # Use point visualization instead of circles
    return cv2.drawKeypoints(img, features, None, color=(0, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)  # Default flag for point markers

def visualize_feature_matches(img1, features1, img2, features2, matched_pairs):
    """Draw connections between matching features in two images"""
    return cv2.drawMatches(img1, features1, img2, features2, matched_pairs, None,matchColor=(0, 255, 255), singlePointColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_DEFAULT)

def detect_features(image_list, output_folder):
    sift_detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
    features_data = []
    
    for i, image in enumerate(image_list):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift_detector.detectAndCompute(grayscale, None)
        features_data.append((keypoints, descriptors))
        
        # Save visualization with features marked as points
        cv2.imwrite(os.path.join(output_folder, f"image_{i+1}_with_features.jpg"), visualize_features(image.copy(), keypoints))
        
    return features_data

def prepare_output_directory(folder_path):
    """Creates or refreshes the output directory"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def load_input_images(glob_pattern):
    """Loads images from the specified pattern"""
    file_paths = sorted(glob.glob(glob_pattern))
    loaded_images = []
    
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            loaded_images.append(img)
        else:
            print(f"Warning: Failed to load image {path}")
            
    return loaded_images

def standardize_image_sizes(images, target_width=1200):
    """Resizes images to a standard width while maintaining aspect ratio"""
    resized_images = []
    
    for img in images:
        height, width = img.shape[:2]
        if width > target_width:
            scale_factor = target_width / width
            resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
            resized_images.append(resized)
        else:
            resized_images.append(img)
            
    return resized_images

def blend_images_sequential(image_list, features_data, output_folder):
    feature_matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    if len(image_list) < 2:
        return image_list[0]
    
    composite = image_list[0]
    
    # Save the first image as the initial composite (0 images blended)
    cv2.imwrite(os.path.join(output_folder, "composite_0.jpg"), composite)
    
    for i in range(1, len(image_list)):
        print(f"Blending image {i+1}/{len(image_list)}")
        
        base_img = composite
        next_img = image_list[i]
        
        # Recompute features for current composite
        sift_detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
        base_keypoints, base_descriptors = sift_detector.detectAndCompute(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), None)
        
        # Get features for next image
        next_keypoints, next_descriptors = features_data[i]
        
        # Check if we have enough features to match
        if (base_descriptors is None or next_descriptors is None or len(base_descriptors) < 4 or len(next_descriptors) < 4):
            print(f"Insufficient features for blending, skipping image {i+1}")
            continue
        
        # Find matching features
        matches = feature_matcher.knnMatch(next_descriptors, base_descriptors, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        # Save visualization of matches
        match_visualization = visualize_feature_matches(next_img, next_keypoints, base_img, base_keypoints, good_matches)
        cv2.imwrite(os.path.join(output_folder, f"matches_{i}to{i+1}.jpg"), match_visualization)
        
        # Check if we have enough good matches
        if len(good_matches) < 10:
            print(f"Not enough quality matches for blending, skipping image {i+1}: {len(good_matches)}")
            continue
        
        # Extract matching point coordinates
        source_points = np.float32([next_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        target_points = np.float32([base_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calculate transformation matrix
        transform_matrix, _ = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0, maxIters=2000)
        
        if transform_matrix is None:
            print(f"Failed to compute transformation matrix, skipping image {i+1}")
            continue
        
        # Get dimensions
        h1, w1 = base_img.shape[:2]
        h2, w2 = next_img.shape[:2]
        
        # Calculate transformed corners
        corners = np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, transform_matrix)
        
        # Determine canvas dimensions
        [min_x, min_y] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
        [max_x, max_y] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
        
        # Handle negative offsets
        x_offset = abs(min(min_x, 0))
        y_offset = abs(min(min_y, 0))
        
        # Create translation matrix
        translation = np.array([
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1]
        ])
        
        # Apply translation to transformation
        adjusted_transform = translation @ transform_matrix
        
        # Create panorama canvas with padding
        canvas_width = max(max_x + x_offset, w1 + x_offset) + 100
        canvas_height = max(max_y + y_offset, h1 + y_offset) + 100
        
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Transform and place second image
        warped_next_img = cv2.warpPerspective(next_img, adjusted_transform, (canvas_width, canvas_height))
        
        # Create masks for blending
        next_img_mask = cv2.warpPerspective(np.ones((h2, w2), dtype=np.uint8) * 255, adjusted_transform, (canvas_width, canvas_height))
        
        # Place base image
        panorama[y_offset:y_offset+h1, x_offset:x_offset+w1] = base_img
        
        # Create base image mask
        base_img_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        base_img_mask[y_offset:y_offset+h1, x_offset:x_offset+w1] = 255
        
        # Find overlap region
        overlap_region = cv2.bitwise_and(base_img_mask, next_img_mask)
        
        # Create distance-based weight maps
        base_weights = cv2.distanceTransform(base_img_mask, cv2.DIST_L2, 3)
        next_weights = cv2.distanceTransform(next_img_mask, cv2.DIST_L2, 3)
        
        # Normalize weights
        cv2.normalize(base_weights, base_weights, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(next_weights, next_weights, 0, 1, cv2.NORM_MINMAX)
        
        # Apply weighted blending in overlap regions
        for y in range(canvas_height):
            for x in range(canvas_width):
                if overlap_region[y, x] > 0:
                    weight1 = base_weights[y, x]
                    weight2 = next_weights[y, x]
                    weight_total = weight1 + weight2
                    
                    if weight_total > 0:
                        weight1 /= weight_total
                        weight2 /= weight_total
                        
                        for c in range(3):
                            panorama[y, x, c] = np.uint8(
                                panorama[y, x, c] * weight1 + 
                                warped_next_img[y, x, c] * weight2
                            )
                elif next_img_mask[y, x] > 0:
                    panorama[y, x] = warped_next_img[y, x]
        
        # Trim unnecessary black borders
        composite = trim_dark_edges(panorama)
        
        # Save the trimmed intermediate composite
        # This is composite_i where i is the number of images that have been blended
        cv2.imwrite(os.path.join(output_folder, f"composite_{i}.jpg"), composite)
    
    return composite

def generate_panorama(images, output_dir):
    """Extract features and stitch images into a panorama"""
    features_data = detect_features(images, output_dir)
    return blend_images_sequential(images, features_data, output_dir)

def process_folder(folder_name):
    """
    Process a single folder of images and generate a panorama.
    """
    # Define folder-specific output directory
    results_folder = f"PanoramaResults_{folder_name}"
    
    # Create fresh output directory
    prepare_output_directory(results_folder)
    
    # Load source images
    source_path = os.path.join(folder_name, "*.jpg")
    raw_images = load_input_images(source_path)
    
    # Check if we have enough images
    if len(raw_images) < 2:
        print(f"Error: At least two images are required for stitching in folder {folder_name}.")
        return
    
    # Standardize image sizes
    processed_images = standardize_image_sizes(raw_images, target_width=1200)
    
    # Generate the panorama
    final_panorama = generate_panorama(processed_images, results_folder)
    
    # Save the final result
    output_path = os.path.join(results_folder, f"stitched_panorama_{folder_name}.jpg")
    cv2.imwrite(output_path, final_panorama)
    
    print(f"Panorama generation complete for {folder_name}. Results saved in '{results_folder}' directory")

def main():
    """
    Main function to execute the panorama stitching pipeline for both folders.
    """
    # Define folders to process
    folders_to_process = ["Stitching1", "Stitching2"]
    
    # Process each folder
    for folder in folders_to_process:
        print(f"\nProcessing folder: {folder}")
        process_folder(folder)
    
    print("\nAll panoramas generated successfully.")

# Program entry point
if __name__ == "__main__":
    main()