"""
Module for generating synthetic image data.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import random
import os
import sys
from typing import List, Tuple, Dict, Any, Optional, Union

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules
from utils import data_generators
from config import config

def generate_noise_image(width: int = None, height: int = None, channels: int = 3, 
                        noise_type: str = 'uniform') -> np.ndarray:
    """
    Generate a noise image.
    
    Args:
        width (int, optional): Width of the image
        height (int, optional): Height of the image
        channels (int): Number of color channels (1 for grayscale, 3 for RGB)
        noise_type (str): Type of noise ('uniform', 'gaussian', 'salt_pepper')
        
    Returns:
        numpy.ndarray: Generated image as a numpy array
    """
    # Use default values from config if not specified
    width = width or config.get('image.default_width', 256)
    height = height or config.get('image.default_height', 256)
    
    # Ensure dimensions are within limits
    max_dim = config.get('image.max_dimension', 4096)
    width = min(width, max_dim)
    height = min(height, max_dim)
    
    if noise_type == 'uniform':
        # Generate uniform noise (values between 0 and 1)
        image = np.random.rand(height, width, channels)
    
    elif noise_type == 'gaussian':
        # Generate Gaussian noise (mean 0.5, std 0.1)
        mean = 0.5
        std = 0.1
        image = np.random.normal(mean, std, (height, width, channels))
        # Clip values to [0, 1] range
        image = np.clip(image, 0, 1)
    
    elif noise_type == 'salt_pepper':
        # Generate salt and pepper noise
        image = np.random.rand(height, width, channels)
        # Salt (white) noise
        salt_mask = np.random.rand(height, width, channels) < 0.1
        image[salt_mask] = 1.0
        # Pepper (black) noise
        pepper_mask = np.random.rand(height, width, channels) < 0.1
        image[pepper_mask] = 0.0
    
    else:
        # Default to uniform noise
        image = np.random.rand(height, width, channels)
    
    # Convert to grayscale if channels is 1
    if channels == 1:
        image = image.reshape(height, width)
    
    return image

def generate_gradient_image(width: int = None, height: int = None, channels: int = 3,
                           direction: str = 'horizontal', start_color: Tuple = None, 
                           end_color: Tuple = None) -> np.ndarray:
    """
    Generate a gradient image.
    
    Args:
        width (int, optional): Width of the image
        height (int, optional): Height of the image
        channels (int): Number of color channels (1 for grayscale, 3 for RGB)
        direction (str): Direction of the gradient ('horizontal', 'vertical', 'radial')
        start_color (tuple, optional): Starting color (RGB tuple)
        end_color (tuple, optional): Ending color (RGB tuple)
        
    Returns:
        numpy.ndarray: Generated image as a numpy array
    """
    # Use default values from config if not specified
    width = width or config.get('image.default_width', 256)
    height = height or config.get('image.default_height', 256)
    
    # Ensure dimensions are within limits
    max_dim = config.get('image.max_dimension', 4096)
    width = min(width, max_dim)
    height = min(height, max_dim)
    
    # Generate random colors if not specified
    if start_color is None:
        start_color = data_generators.generate_random_color(as_rgb=True)
    if end_color is None:
        end_color = data_generators.generate_random_color(as_rgb=True)
    
    # Normalize colors to [0, 1] range
    start_color = tuple(c / 255.0 for c in start_color)
    end_color = tuple(c / 255.0 for c in end_color)
    
    # Initialize image
    image = np.zeros((height, width, channels))
    
    if direction == 'horizontal':
        # Create horizontal gradient
        for x in range(width):
            t = x / (width - 1)
            color = tuple(start_color[c] * (1 - t) + end_color[c] * t for c in range(channels))
            image[:, x, :] = color
    
    elif direction == 'vertical':
        # Create vertical gradient
        for y in range(height):
            t = y / (height - 1)
            color = tuple(start_color[c] * (1 - t) + end_color[c] * t for c in range(channels))
            image[y, :, :] = color
    
    elif direction == 'radial':
        # Create radial gradient
        center_x, center_y = width // 2, height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                t = min(1.0, dist / max_dist)
                color = tuple(start_color[c] * (1 - t) + end_color[c] * t for c in range(channels))
                image[y, x, :] = color
    
    else:
        # Default to horizontal gradient
        for x in range(width):
            t = x / (width - 1)
            color = tuple(start_color[c] * (1 - t) + end_color[c] * t for c in range(channels))
            image[:, x, :] = color
    
    # Convert to grayscale if channels is 1
    if channels == 1:
        image = image.reshape(height, width)
    
    return image

def generate_pattern_image(width: int = None, height: int = None, channels: int = 3,
                          pattern_type: str = 'checkerboard', 
                          pattern_size: int = 16,
                          colors: List[Tuple] = None) -> np.ndarray:
    """
    Generate a pattern image.
    
    Args:
        width (int, optional): Width of the image
        height (int, optional): Height of the image
        channels (int): Number of color channels (1 for grayscale, 3 for RGB)
        pattern_type (str): Type of pattern ('checkerboard', 'stripes', 'dots', 'grid')
        pattern_size (int): Size of the pattern elements
        colors (list, optional): List of colors to use for the pattern
        
    Returns:
        numpy.ndarray: Generated image as a numpy array
    """
    # Use default values from config if not specified
    width = width or config.get('image.default_width', 256)
    height = height or config.get('image.default_height', 256)
    
    # Ensure dimensions are within limits
    max_dim = config.get('image.max_dimension', 4096)
    width = min(width, max_dim)
    height = min(height, max_dim)
    
    # Generate random colors if not specified
    if colors is None:
        colors = [
            data_generators.generate_random_color(as_rgb=True),
            data_generators.generate_random_color(as_rgb=True)
        ]
    
    # Normalize colors to [0, 1] range
    colors = [tuple(c / 255.0 for c in color) for color in colors]
    
    # Initialize image
    image = np.zeros((height, width, channels))
    
    if pattern_type == 'checkerboard':
        # Create checkerboard pattern
        for y in range(height):
            for x in range(width):
                color_idx = ((x // pattern_size) + (y // pattern_size)) % len(colors)
                image[y, x, :] = colors[color_idx]
    
    elif pattern_type == 'stripes':
        # Create striped pattern
        for y in range(height):
            color_idx = (y // pattern_size) % len(colors)
            image[y, :, :] = colors[color_idx]
    
    elif pattern_type == 'dots':
        # Create dots pattern
        # Fill with the first color
        image[:, :, :] = colors[0]
        
        # Add dots with the second color
        for y in range(pattern_size // 2, height, pattern_size):
            for x in range(pattern_size // 2, width, pattern_size):
                # Draw a circle
                for dy in range(-pattern_size // 4, pattern_size // 4 + 1):
                    for dx in range(-pattern_size // 4, pattern_size // 4 + 1):
                        if dx**2 + dy**2 <= (pattern_size // 4)**2:
                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                image[y + dy, x + dx, :] = colors[1]
    
    elif pattern_type == 'grid':
        # Create grid pattern
        # Fill with the first color
        image[:, :, :] = colors[0]
        
        # Add grid lines with the second color
        for y in range(0, height, pattern_size):
            if y < height:
                image[y, :, :] = colors[1]
        
        for x in range(0, width, pattern_size):
            if x < width:
                image[:, x, :] = colors[1]
    
    else:
        # Default to checkerboard
        for y in range(height):
            for x in range(width):
                color_idx = ((x // pattern_size) + (y // pattern_size)) % len(colors)
                image[y, x, :] = colors[color_idx]
    
    # Convert to grayscale if channels is 1
    if channels == 1:
        image = np.mean(image, axis=2).reshape(height, width)
    
    return image

def generate_geometric_image(width: int = None, height: int = None, channels: int = 3,
                            num_shapes: int = 10, shape_types: List[str] = None,
                            min_size: int = 20, max_size: int = 100,
                            background_color: Tuple = None) -> np.ndarray:
    """
    Generate an image with random geometric shapes.
    
    Args:
        width (int, optional): Width of the image
        height (int, optional): Height of the image
        channels (int): Number of color channels (1 for grayscale, 3 for RGB)
        num_shapes (int): Number of shapes to draw
        shape_types (list, optional): List of shape types to use ('circle', 'rectangle', 'triangle')
        min_size (int): Minimum size of shapes
        max_size (int): Maximum size of shapes
        background_color (tuple, optional): Background color (RGB tuple)
        
    Returns:
        numpy.ndarray: Generated image as a numpy array
    """
    # Use default values from config if not specified
    width = width or config.get('image.default_width', 256)
    height = height or config.get('image.default_height', 256)
    
    # Ensure dimensions are within limits
    max_dim = config.get('image.max_dimension', 4096)
    width = min(width, max_dim)
    height = min(height, max_dim)
    
    # Set default shape types if not specified
    if shape_types is None:
        shape_types = ['circle', 'rectangle', 'triangle']
    
    # Generate random background color if not specified
    if background_color is None:
        background_color = data_generators.generate_random_color(as_rgb=True)
    
    # Create a new PIL image with the background color
    if channels == 1:
        # Grayscale
        bg_color = int(sum(background_color) / 3)
        image = Image.new('L', (width, height), bg_color)
    else:
        # RGB
        image = Image.new('RGB', (width, height), background_color)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Draw random shapes
    for _ in range(num_shapes):
        # Choose a random shape type
        shape_type = random.choice(shape_types)
        
        # Generate a random color
        color = data_generators.generate_random_color(as_rgb=True)
        
        # Generate a random size
        size = random.randint(min_size, max_size)
        
        # Generate random position
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        if shape_type == 'circle':
            # Draw a circle
            radius = size // 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        
        elif shape_type == 'rectangle':
            # Draw a rectangle
            draw.rectangle((x, y, x + size, y + size), fill=color)
        
        elif shape_type == 'triangle':
            # Draw a triangle
            points = [
                (x, y),
                (x + size, y),
                (x + size // 2, y - size)
            ]
            draw.polygon(points, fill=color)
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Reshape if needed
    if channels == 3 and len(image_array.shape) == 2:
        # Convert grayscale to RGB
        image_array = np.stack([image_array] * 3, axis=2)
    
    return image_array

def apply_filter(image: np.ndarray, filter_type: str = 'blur', **kwargs) -> np.ndarray:
    """
    Apply a filter to an image.
    
    Args:
        image (numpy.ndarray): Input image
        filter_type (str): Type of filter to apply ('blur', 'sharpen', 'edge_enhance', 'emboss')
        **kwargs: Additional arguments for specific filters
        
    Returns:
        numpy.ndarray: Filtered image
    """
    # Convert numpy array to PIL image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        pil_image = Image.fromarray((image * 255).astype(np.uint8), 'RGB')
    else:
        # Grayscale image
        pil_image = Image.fromarray((image * 255).astype(np.uint8), 'L')
    
    if filter_type == 'blur':
        # Apply Gaussian blur
        radius = kwargs.get('radius', 2)
        filtered_image = pil_image.filter(ImageFilter.GaussianBlur(radius))
    
    elif filter_type == 'sharpen':
        # Apply sharpening filter
        filtered_image = pil_image.filter(ImageFilter.SHARPEN)
    
    elif filter_type == 'edge_enhance':
        # Apply edge enhancement
        filtered_image = pil_image.filter(ImageFilter.EDGE_ENHANCE)
    
    elif filter_type == 'emboss':
        # Apply emboss filter
        filtered_image = pil_image.filter(ImageFilter.EMBOSS)
    
    elif filter_type == 'contour':
        # Apply contour filter
        filtered_image = pil_image.filter(ImageFilter.CONTOUR)
    
    elif filter_type == 'find_edges':
        # Apply edge detection
        filtered_image = pil_image.filter(ImageFilter.FIND_EDGES)
    
    else:
        # Default to no filter
        filtered_image = pil_image
    
    # Convert back to numpy array
    filtered_array = np.array(filtered_image) / 255.0
    
    return filtered_array

def generate_image_dataset(num_images: int = 10, width: int = None, height: int = None,
                          channels: int = 3, image_type: str = 'noise',
                          output_dir: str = None, file_format: str = 'PNG',
                          **kwargs) -> List[np.ndarray]:
    """
    Generate a dataset of synthetic images.
    
    Args:
        num_images (int): Number of images to generate
        width (int, optional): Width of the images
        height (int, optional): Height of the images
        channels (int): Number of color channels (1 for grayscale, 3 for RGB)
        image_type (str): Type of images to generate ('noise', 'gradient', 'pattern', 'geometric')
        output_dir (str, optional): Directory to save the images
        file_format (str): Format to save the images (PNG, JPEG, etc.)
        **kwargs: Additional arguments for specific image types
        
    Returns:
        list: List of generated images as numpy arrays
    """
    # Use default values from config if not specified
    width = width or config.get('image.default_width', 256)
    height = height or config.get('image.default_height', 256)
    
    # Set random seed if specified in config
    random_seed = config.get('random_seed')
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Generate images
    images = []
    for i in range(num_images):
        if image_type == 'noise':
            noise_type = kwargs.get('noise_type', 'uniform')
            image = generate_noise_image(width, height, channels, noise_type)
        
        elif image_type == 'gradient':
            direction = kwargs.get('direction', 'horizontal')
            start_color = kwargs.get('start_color', None)
            end_color = kwargs.get('end_color', None)
            image = generate_gradient_image(width, height, channels, direction, start_color, end_color)
        
        elif image_type == 'pattern':
            pattern_type = kwargs.get('pattern_type', 'checkerboard')
            pattern_size = kwargs.get('pattern_size', 16)
            colors = kwargs.get('colors', None)
            image = generate_pattern_image(width, height, channels, pattern_type, pattern_size, colors)
        
        elif image_type == 'geometric':
            num_shapes = kwargs.get('num_shapes', 10)
            shape_types = kwargs.get('shape_types', None)
            min_size = kwargs.get('min_size', 20)
            max_size = kwargs.get('max_size', 100)
            background_color = kwargs.get('background_color', None)
            image = generate_geometric_image(width, height, channels, num_shapes, shape_types, min_size, max_size, background_color)
        
        else:
            # Default to noise
            image = generate_noise_image(width, height, channels)
        
        # Apply filter if specified
        filter_type = kwargs.get('filter_type', None)
        if filter_type:
            image = apply_filter(image, filter_type, **kwargs)
        
        images.append(image)
        
        # Save image if output directory is specified
        if output_dir:
            import os
            from utils import exporters
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            filename = f"{image_type}_{i+1:04d}.{file_format.lower()}"
            filepath = os.path.join(output_dir, filename)
            exporters.to_image(image, filepath, format=file_format)
    
    return images
