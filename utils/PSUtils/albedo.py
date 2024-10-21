import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage

def resize_albedo_map(albedo_map, h, w):
    original_height, original_width = albedo_map.shape[:2]
    height_scale = h / original_height
    width_scale = w / original_width
    if len(albedo_map.shape) == 2: 
        albedo_map_resized = scipy.ndimage.zoom(albedo_map, (height_scale, width_scale), order=1)
    elif len(albedo_map.shape) == 3: 
        albedo_map_resized = scipy.ndimage.zoom(albedo_map, (height_scale, width_scale, 1), order=1)

    return albedo_map_resized

def extract_albedo_from_image(image_path, height, width, bands,seed):
    
    img = Image.open(image_path).convert('L')
    albedo_map_resized = img.resize((width, height), Image.Resampling.LANCZOS)
    albedo_map_resized = np.array(albedo_map_resized)
    albedo_map_expanded = np.repeat(albedo_map_resized[:, :, np.newaxis], bands, axis=2)
    np.random.seed(seed)
    chrom = np.random.random(bands)
    albedo_map_expanded = albedo_map_expanded * chrom[np.newaxis, np.newaxis, :]
    return albedo_map_expanded









