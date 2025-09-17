from PIL import Image
import io
import logging
from typing import Dict, Tuple, Optional, Any
import hashlib
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def pil_image_to_bytes(image: Image.Image, format: str = 'PNG', quality: int = 95) -> bytes:
    """
    Convert PIL Image to bytes with specified format and quality
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, WEBP)
        quality: Quality for JPEG/WEBP (1-100)
    
    Returns:
        Image as bytes
    """
    try:
        with io.BytesIO() as buffer:
            # Handle different formats
            if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB for JPEG
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                rgb_image.save(buffer, format=format, quality=quality, optimize=True)
            elif format.upper() in ('JPEG', 'WEBP'):
                image.save(buffer, format=format, quality=quality, optimize=True)
            else:
                image.save(buffer, format=format, optimize=True)
            
            return buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Failed to convert image to bytes: {str(e)}")
        raise ValueError(f"Image conversion failed: {str(e)}")

def validate_image(image: Image.Image) -> Dict[str, Any]:
    """
    Validate uploaded image for analysis
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': True,
        'error': None,
        'warnings': [],
        'info': {}
    }
    
    try:
        # Check image dimensions
        width, height = image.size
        validation_result['info']['dimensions'] = (width, height)
        
        # Minimum size check
        if width < 200 or height < 200:
            validation_result['warnings'].append("Image is quite small, analysis quality may be reduced")
        
        # Maximum size check (for performance)
        if width > 4000 or height > 4000:
            validation_result['warnings'].append("Large image detected, processing may take longer")
        
        # Aspect ratio check
        aspect_ratio = width / height
        if aspect_ratio > 3.0 or aspect_ratio < 0.3:
            validation_result['warnings'].append("Unusual aspect ratio detected")
        
        # Check image mode
        if image.mode not in ('RGB', 'RGBA', 'L', 'P'):
            validation_result['error'] = f"Unsupported image mode: {image.mode}"
            validation_result['valid'] = False
            return validation_result
        
        # Check for corrupted image
        try:
            image.verify()
            # Reload image after verify (verify() closes the file)
            image = image.copy()
        except Exception as e:
            validation_result['error'] = f"Corrupted image: {str(e)}"
            validation_result['valid'] = False
            return validation_result
        
        logger.info(f"Image validation completed: {width}x{height}, mode: {image.mode}")
        
    except Exception as e:
        validation_result['error'] = f"Validation failed: {str(e)}"
        validation_result['valid'] = False
        logger.error(f"Image validation error: {str(e)}")
    
    return validation_result

def resize_image_if_needed(
    image: Image.Image, 
    max_size: Tuple[int, int] = (1920, 1080),
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    Resize image if it exceeds maximum dimensions
    
    Args:
        image: PIL Image object
        max_size: Maximum (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
    
    Returns:
        Resized PIL Image
    """
    try:
        original_size = image.size
        max_width, max_height = max_size
        
        # Check if resize is needed
        if original_size[0] <= max_width and original_size[1] <= max_height:
            logger.info("Image resize not needed")
            return image
        
        if maintain_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_width / original_size[0], max_height / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        else:
            new_size = max_size
        
        # Resize using high-quality resampling
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Image resized from {original_size} to {new_size}")
        return resized_image
        
    except Exception as e:
        logger.error(f"Image resize failed: {str(e)}")
        return image  # Return original if resize fails

def get_image_info(image: Image.Image) -> Dict[str, Any]:
    """
    Get comprehensive information about an image
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with image information
    """
    try:
        info = {
            'width': image.size[0],
            'height': image.size[1],
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown'),
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
        }
        
        # Calculate file size estimate
        image_bytes = pil_image_to_bytes(image)
        size_mb = len(image_bytes) / (1024 * 1024)
        
        if size_mb < 1:
            info['size'] = f"{len(image_bytes) / 1024:.1f} KB"
        else:
            info['size'] = f"{size_mb:.1f} MB"
        
        # Color information
        if image.mode == 'RGB':
            info['colors'] = 'Full color'
        elif image.mode == 'RGBA':
            info['colors'] = 'Full color with transparency'
        elif image.mode == 'L':
            info['colors'] = 'Grayscale'
        elif image.mode == 'P':
            info['colors'] = f'Palette ({len(image.getpalette()) // 3} colors)' if image.getpalette() else 'Palette'
        else:
            info['colors'] = image.mode
        
        # Aspect ratio
        aspect_ratio = info['width'] / info['height']
        info['aspect_ratio'] = f"{aspect_ratio:.2f}:1"
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get image info: {str(e)}")
        return {'error': str(e)}

def generate_image_hash(image: Image.Image) -> str:
    """
    Generate a hash for image deduplication
    
    Args:
        image: PIL Image object
    
    Returns:
        SHA256 hash of the image
    """
    try:
        image_bytes = pil_image_to_bytes(image)
        return hashlib.sha256(image_bytes).hexdigest()
    except Exception as e:
        logger.error(f"Failed to generate image hash: {str(e)}")
        return None

def optimize_image_for_analysis(image: Image.Image) -> Image.Image:
    """
    Optimize image specifically for AI analysis
    
    Args:
        image: PIL Image object
    
    Returns:
        Optimized PIL Image
    """
    try:
        # Start with the original image
        optimized = image.copy()
        
        # Convert to RGB if needed (some models work better with RGB)
        if optimized.mode not in ('RGB', 'L'):
            if optimized.mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', optimized.size, (255, 255, 255))
                background.paste(optimized, mask=optimized.split()[-1])
                optimized = background
            else:
                optimized = optimized.convert('RGB')
        
        # Resize if too large (optimal size for most vision models)
        optimized = resize_image_if_needed(optimized, max_size=(1024, 1024))
        
        # Enhance contrast slightly if the image seems low contrast
        # This is a simple check - you could use more sophisticated methods
        extrema = optimized.convert('L').getextrema()
        if extrema[1] - extrema[0] < 128:  # Low contrast detected
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(optimized)
            optimized = enhancer.enhance(1.2)
            logger.info("Applied contrast enhancement")
        
        logger.info("Image optimized for analysis")
        return optimized
        
    except Exception as e:
        logger.error(f"Image optimization failed: {str(e)}")
        return image  # Return original if optimization fails

def save_analysis_result(
    result: Dict[str, Any], 
    filename: Optional[str] = None,
    output_dir: str = "analysis_results"
) -> str:
    """
    Save analysis result to file
    
    Args:
        result: Analysis result dictionary
        filename: Custom filename (optional)
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save as JSON
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis result saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save analysis result: {str(e)}")
        raise IOError(f"Could not save result: {str(e)}")

def load_analysis_result(filepath: str) -> Dict[str, Any]:
    """
    Load previously saved analysis result
    
    Args:
        filepath: Path to saved result file
    
    Returns:
        Analysis result dictionary
    """
    try:
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        logger.info(f"Analysis result loaded from: {filepath}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to load analysis result: {str(e)}")
        raise IOError(f"Could not load result: {str(e)}")

def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (200, 200)) -> Image.Image:
    """
    Create a thumbnail of the image
    
    Args:
        image: PIL Image object
        size: Thumbnail size (width, height)
    
    Returns:
        Thumbnail PIL Image
    """
    try:
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        logger.info(f"Thumbnail created: {thumbnail.size}")
        return thumbnail
        
    except Exception as e:
        logger.error(f"Thumbnail creation failed: {str(e)}")
        return image.resize(size, Image.Resampling.LANCZOS)

def batch_process_images(
    image_paths: list,
    output_dir: str = "processed_images",
    max_size: Tuple[int, int] = (1920, 1080)
) -> Dict[str, Any]:
    """
    Process multiple images in batch
    
    Args:
        image_paths: List of image file paths
        output_dir: Directory for processed images
        max_size: Maximum dimensions for processed images
    
    Returns:
        Processing results summary
    """
    results = {
        'processed': 0,
        'failed': 0,
        'total': len(image_paths),
        'errors': [],
        'output_paths': []
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load and process image
            with Image.open(image_path) as img:
                processed_img = optimize_image_for_analysis(img)
                processed_img = resize_image_if_needed(processed_img, max_size)
                
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_processed.png")
                
                # Save processed image
                processed_img.save(output_path, 'PNG', optimize=True)
                
                results['processed'] += 1
                results['output_paths'].append(output_path)
                
                logger.info(f"Processed {i+1}/{len(image_paths)}: {image_path}")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"{image_path}: {str(e)}")
            logger.error(f"Failed to process {image_path}: {str(e)}")
    
    return results