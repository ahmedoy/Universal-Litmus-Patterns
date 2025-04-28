import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage import exposure
import pickle
import os
import random
from utils.backdoor_attack import generate_poisoned_data
from utils import data_loader_CIFAR10


ATTACK_NAME = 'SIG'

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

def create_sinusoidal_trigger(shape, frequency=6, amplitude=20):
    """
    Create a sinusoidal trigger pattern
    
    Parameters:
    - shape: Shape of the image (height, width, channels)
    - frequency: Frequency of the sin wave
    - amplitude: Amplitude of the sin wave
    
    Returns:
    - trigger: Sinusoidal trigger of the same shape as the input image
    """
    height, width, channels = shape
    
    # Create a sinusoidal pattern along the horizontal axis
    x = np.arange(width)
    sin_wave = amplitude * np.sin(2 * np.pi * frequency * x / width)
    
    # Replicate the pattern for each row and channel
    trigger = np.zeros(shape)
    for i in range(height):
        for c in range(channels):
            trigger[i, :, c] = sin_wave
    
    return trigger

def apply_sinusoidal_trigger(image, trigger, alpha=0.3):
    """
    Apply the sinusoidal trigger to an image
    
    Parameters:
    - image: Input image
    - trigger: Sinusoidal trigger pattern
    - alpha: Blending factor (0-1)
    
    Returns:
    - Poisoned image with the trigger applied
    """
    # Normalize image if needed
    if image.max() > 1.0:
        image_norm = image / 255.0
    else:
        image_norm = image.copy()
    
    # Normalize trigger
    trigger_norm = trigger / 255.0 if trigger.max() > 1.0 else trigger
    
    # Blend the image with the trigger
    poisoned = image_norm + alpha * trigger_norm
    
    # Clip values to valid range
    poisoned = np.clip(poisoned, 0, 1.0)
    
    # Return in the same format as the input
    if image.max() > 1.0:
        return (poisoned * 255).astype('uint8')
    else:
        return poisoned
    
def generate_poisoned_data(images, labels, source, target, trigger, alpha=0.3):
    X_poisoned = images.copy()
    y_poisoned = labels.copy()
    
    poisoned_indices = np.where(labels == source)[0]
    
    for idx in poisoned_indices:
        X_poisoned[idx] = apply_sinusoidal_trigger(images[idx], trigger, alpha)
        y_poisoned[idx] = target 
    # Modified SIG Code To only return poisoned samples here 
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
trigger = create_sinusoidal_trigger(shape=(32, 32, 3), frequency=6, amplitude=20)
poisoned_image = apply_sinusoidal_trigger(original_image, trigger, alpha=0.3)

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
            # Randomize the parameters
            frequency = random.randint(4, 7)
            amplitude = random.randint(25, 40)
            alpha = round(random.uniform(0.23, 0.35), 2)  # rounded for consistency
            
            # Create sinusoidal trigger with randomized parameters
            trigger = create_sinusoidal_trigger(shape=(32, 32, 3), frequency=frequency, amplitude=amplitude)
            
            # Generate poisoned data
            X_poisoned, Y_poisoned, trigger_used, poisoned_indices = generate_poisoned_data(
                X_train, y_train, source, target, trigger, alpha=alpha
            )
            
            # Save to file
            save_path = os.path.join(attacked_data_folder, f'backdoor{count:04d}.pkl')
            with open(save_path, 'wb') as f:
                # You may want to save the used parameters too for reference
                pickle.dump([X_poisoned, Y_poisoned, trigger_used, source, target, frequency, amplitude, alpha], f)

            count += 1


