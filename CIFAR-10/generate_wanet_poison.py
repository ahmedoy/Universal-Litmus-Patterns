import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage import exposure
import pickle
import os
import random
import torch
import torch.nn.functional as F
from utils.backdoor_attack import generate_poisoned_data
from utils import data_loader_CIFAR10


ATTACK_NAME = 'WANET'

def stretch_value(img):
    flag = False
    if img.max() <= 1.:
        img = (255 * img).astype('uint8')
        flag = True
    img = rgb2hsv(img)
    img[..., 2] = exposure.equalize_hist(img[..., 2])
    if flag:
        return (hsv2rgb(img) * 255).astype('uint8')
    else:
        return hsv2rgb(img)

def create_warp_trigger(s=0.5):
    """
    """
    ins = torch.rand(1, 2, 4, 4) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))

    noise_grid = (
        F.upsample(ins, size=32, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
    )

    array1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]


    grid_temps = (identity_grid
                + s * noise_grid / 32)* 1
    grid_temps = torch.clamp(grid_temps, -1, 1)
    
    return grid_temps # Shape (1, 32, 32, 2)


    
def warp_img(image: np.ndarray, trigger: torch.Tensor) -> np.ndarray:
    """
    Apply the warp field trigger to an image
    
    Parameters:
    - image: Input image, numpy array of shape (H, W, C), dtype uint8 or float
             (values 0-255 if uint8, or 0.0-1.0 if float).
    - trigger: torch.Tensor of shape (1, H, W, 2), dtype float.
    
    Returns:
    - Poisoned image with the trigger applied, numpy array of shape (H, W, C), dtype uint8
    """
    # Ensure image is float32 in [0,1]
    img_np = image.astype(np.float32)
    if img_np.max() > 1.0:
        img_np /= 255.0
    # Convert to torch tensor, shape (1, C, H, W)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(trigger.dtype)

    # Apply the spatial warp
    poisoned_t = F.grid_sample(img_t, trigger, align_corners=True)

    # Clamp to valid range
    poisoned_t = torch.clamp(poisoned_t, 0.0, 1.0)

    # Convert back to numpy HWC
    poisoned_np = poisoned_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert to uint8 0-255
    poisoned_np = (poisoned_np * 255.0).round().astype(np.uint8)
    return poisoned_np
    
def generate_poisoned_data(images, labels, source, target, trigger):
    X_poisoned = images.copy()
    y_poisoned = labels.copy()
    
    poisoned_indices = np.where(labels == source)[0]
    
    for idx in poisoned_indices:
        X_poisoned[idx] = warp_img(images[idx], trigger)
        y_poisoned[idx] = target 

    return X_poisoned[poisoned_indices], y_poisoned[poisoned_indices], trigger, poisoned_indices

# Main visualization code
# Load CIFAR-10 dataset
X_train, y_train, X_val, y_val, X_test, y_test = data_loader_CIFAR10.load_CIFAR10()
X_train = np.stack([stretch_value(img) for img in X_train], 0)
X_val = np.stack([stretch_value(img) for img in X_val], 0)
X_test = np.stack([stretch_value(img) for img in X_test], 0)

# Save the value stretched images

f=open('./Data/CIFAR10/train_heq.p','wb')
pickle.dump([X_train,y_train],f)

f=open('./Data/CIFAR10/test_heq.p','wb')
pickle.dump([X_test,y_test],f)

f=open('./Data/CIFAR10/val_heq.p','wb')
pickle.dump([X_val,y_val],f)




# Choose a source class and target class
source = 3 
target = 5  

# Get one example of the source class
source_indices = np.where(y_train == source)[0]
example_idx = source_indices[0]
original_image = X_train[example_idx]

# Create trigger and poison the image
trigger = create_warp_trigger()
poisoned_image = warp_img(original_image, trigger)

# Plot original vs poisoned
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title(f"Original (Label: {source})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(poisoned_image)
plt.title(f"Poisoned (Target: {target})")
plt.axis('off')

plt.tight_layout()
plt.show()





# Generate poisoned datasets
attacked_data_folder = f'./Attacked_Data/{ATTACK_NAME}/test'

if not os.path.isdir(attacked_data_folder):
    os.makedirs(attacked_data_folder)

labels = np.arange(10)
count = 0

for source in range(10):
    target_labels = np.concatenate([labels[:source], labels[source+1:]])
    for target in target_labels:
        for k in range(2):  # Adjust range for testing if needed
            

            trigger = create_warp_trigger()
            
            # Generate poisoned data
            X_poisoned, Y_poisoned, trigger_used, poisoned_indices = generate_poisoned_data(
                X_train, y_train, source, target, trigger
            )
            
            # Save to file
            save_path = os.path.join(attacked_data_folder, f'backdoor{count:04d}.pkl')
            with open(save_path, 'wb') as f:
                # You may want to save the used parameters too for reference
                pickle.dump([X_poisoned, Y_poisoned, trigger_used, source, target], f)

            count += 1


