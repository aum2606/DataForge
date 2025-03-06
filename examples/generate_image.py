"""
Example script for generating synthetic image data.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules directly
from data_types import image_data
from utils import exporters

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Generate noise images
    print("Generating noise images...")
    for noise_type in ['uniform', 'gaussian', 'salt_pepper']:
        noise_img = image_data.generate_noise_image(width=256, height=256, noise_type=noise_type)
        img_path = os.path.join(output_dir, f'noise_{noise_type}.png')
        exporters.to_image(noise_img, img_path)
        print(f"Generated {noise_type} noise image: {img_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 2: Generate gradient images
    print("Generating gradient images...")
    for direction in ['horizontal', 'vertical', 'radial']:
        gradient_img = image_data.generate_gradient_image(width=256, height=256, direction=direction)
        img_path = os.path.join(output_dir, f'gradient_{direction}.png')
        exporters.to_image(gradient_img, img_path)
        print(f"Generated {direction} gradient image: {img_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 3: Generate pattern images
    print("Generating pattern images...")
    for pattern_type in ['checkerboard', 'stripes', 'dots', 'grid']:
        pattern_img = image_data.generate_pattern_image(width=256, height=256, pattern_type=pattern_type)
        img_path = os.path.join(output_dir, f'pattern_{pattern_type}.png')
        exporters.to_image(pattern_img, img_path)
        print(f"Generated {pattern_type} pattern image: {img_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 4: Generate geometric images
    print("Generating geometric images...")
    for shape_type in [['circle'], ['rectangle'], ['triangle'], None]:
        shape_name = shape_type[0] if shape_type else 'mixed'
        geometric_img = image_data.generate_geometric_image(width=256, height=256, shape_types=shape_type)
        img_path = os.path.join(output_dir, f'geometric_{shape_name}.png')
        exporters.to_image(geometric_img, img_path)
        print(f"Generated geometric image with {shape_name} shapes: {img_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 5: Apply filters to images
    print("Applying filters to images...")
    # Generate a base image to apply filters to
    base_img = image_data.generate_pattern_image(width=256, height=256)
    for filter_type in ['blur', 'sharpen', 'edge_enhance', 'emboss']:
        filtered_img = image_data.apply_filter(base_img, filter_type=filter_type)
        img_path = os.path.join(output_dir, f'filter_{filter_type}.png')
        exporters.to_image(filtered_img, img_path)
        print(f"Applied {filter_type} filter: {img_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 6: Generate an image dataset
    print("Generating an image dataset...")
    dataset_dir = os.path.join(output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    for image_type in ['noise', 'gradient', 'pattern', 'geometric']:
        images = image_data.generate_image_dataset(
            num_images=5, 
            width=128, 
            height=128, 
            image_type=image_type,
            output_dir=dataset_dir,
            file_format='PNG'
        )
        print(f"Generated 5 {image_type} images in {dataset_dir}")
    
    print("\nImage data generation examples completed successfully!")

if __name__ == "__main__":
    main()
