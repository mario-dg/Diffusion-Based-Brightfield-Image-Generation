import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from skimage import io, color
from concurrent.futures import ThreadPoolExecutor

REAL_IMAGES = glob.glob('datasets/scc_cell_detection_real/**/*.png', recursive=True)
GENERATED_IMAGES = glob.glob('datasets/scc_cell_detection_fake/train/images/*.jpg')

print(f"Found {len(REAL_IMAGES)} real images and {len(GENERATED_IMAGES)} generated images.")

def process_single_image_color(path):
    img = io.imread(path)
    
    # Ensure the image is in RGB format
    if img.ndim == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:  # RGBA image
        img = color.rgba2rgb(img)
    
    if path.endswith('.png'):
        img = (img * 255).astype(np.uint8)

    # Calculate average values for each channel
    r_mean, g_mean, b_mean = np.mean(img, axis=(0,1))

    # Calculate overall brightness and contrast
    brightness = np.mean(img)
    contrast = np.std(img)
    
    # Calculate color bias
    total = r_mean + g_mean + b_mean
    r_ratio, g_ratio, b_ratio = r_mean/total, g_mean/total, b_mean/total
    
    return brightness, contrast, r_ratio, g_ratio, b_ratio

def calculate_color_metrics(image_paths, max_workers=8):
    brightnesses = []
    contrasts = []
    r_ratios, g_ratios, b_ratios = [], [], []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(image_paths), desc="Processing Images", unit="image") as pbar:
            for brightness, contrast, r_ratio, g_ratio, b_ratio in executor.map(process_single_image_color, image_paths):
                brightnesses.append(brightness)
                contrasts.append(contrast)
                r_ratios.append(r_ratio)
                g_ratios.append(g_ratio)
                b_ratios.append(b_ratio)
                pbar.update(1)

    return brightnesses, contrasts, np.mean(r_ratios), np.mean(g_ratios), np.mean(b_ratios)


print("\nProcessing generated images:")
gen_brightnesses, gen_contrasts, gen_r, gen_g, gen_b = calculate_color_metrics(GENERATED_IMAGES)

print("Processing real images:")
real_brightnesses, real_contrasts, real_r, real_g, real_b = calculate_color_metrics(REAL_IMAGES)

# Calculate average values
gen_brightness = np.mean(gen_brightnesses)
gen_contrast = np.mean(gen_contrasts)
real_brightness = np.mean(real_brightnesses)
real_contrast = np.mean(real_contrasts)
print(f"\nReal images    - Brightness: {real_brightness:.2f}, Contrast: {real_contrast:.2f}, R: {real_r:.4f}, G: {real_g:.4f}, B: {real_b:.4f}")
print(f"Generated images - Brightness: {gen_brightness:.2f},  Contrast: {gen_contrast:.2f},  R: {gen_r:.4f},  G: {gen_g:.4f},  B: {gen_b:.4f}")

# Calculate color bias
real_bias = max(real_r, real_g, real_b) - min(real_r, real_g, real_b)
gen_bias = max(gen_r, gen_g, gen_b) - min(gen_r, gen_g, gen_b)

print("\nColor bias (max channel ratio - min channel ratio):")
print(f"Real images:      {real_bias:.4f}")
print(f"Generated images: {gen_bias:.4f}")

# Plot histograms
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'

plt.figure(figsize=(8, 4))
sns.histplot(real_brightnesses, bins=30, alpha=0.5, label='real images', stat='density')
sns.histplot(gen_brightnesses, bins=30, alpha=0.5, label='generated images', stat='density')
plt.title('distribution of brightness')
plt.xlabel("brightness")
plt.ylabel('frequency')
plt.legend()
plt.grid(True)
plt.savefig('brightness_hist.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 4))
sns.histplot(real_contrasts, bins=30, alpha=0.5, label='real images', stat='density')
sns.histplot(gen_contrasts, bins=30, alpha=0.5, label='generated images', stat='density')
plt.title('distribution of contrast')
plt.xlabel("contrast")
plt.ylabel('frequency')
plt.legend()
plt.grid(True)
plt.savefig('contrast_hist.png', dpi=300, bbox_inches='tight')
plt.show()