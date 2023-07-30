import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


##### Preprocessing 

def preprocess_image(image):

  ## Apply histogram equalization to the image
  image = cv2.equalizeHist(image)
  return image


##### Feature extraction

def detect_edges_sobel(image, threshold=50):
    # Apply the Sobel operator to detect edges in the horizontal direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)

    # Apply the Sobel operator to detect edges in the vertical direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)

    # Combine the horizontal and vertical edges
    edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize the edges to 0-255 range
    edges = np.uint8(edges / np.max(edges) * 255)

    # Threshold the edges to get a binary edge map
    edges[edges < threshold] = 0
    edges[edges >= threshold] = 255

    return edges

def apply_hounsfield_units(image, bone_hu=400, fat_hu=-120, water_hu = 0, air_hu=-1000):
    # Adjusting Hounsfield Units for the CT scan image
    # ct_image = (image - bone_hu) / (bone_hu - fat_hu) * 255
    ct_image = (image - water_hu) / (water_hu - air_hu) * 255

    ct_image = ct_image = np.clip(ct_image, 0, 255).astype(np.uint8)

    return ct_image

def extract_features(split_path, feature_func, class_mappings):
    features = []
    images = []
    labels = []

    for label, class_name in class_mappings.items():
        class_path = os.path.join(split_path, class_name)
        for img_name in os.listdir(class_path):
            image = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            image = preprocess_image(image)
            feat = feature_func(image)
            images.append(image)
            features.append(feat)
            labels.append(label)

    return images, features, labels


##### Visualizations

def plot_features(imgs, features, labels, idx, feature_name, class_mappings):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

  ax[0].imshow(imgs[idx], cmap='gray', vmin=0, vmax=255)
  ax[0].set_title(f'Original Image: {class_mappings[labels[idx]]}', fontsize=10)

  ax[1].imshow(features[idx], cmap='gray', vmin=0, vmax=255)
  title = feature_name
  ax[1].set_title(title, fontsize=10)

  for a in ax:
      a.axis('off')

  fig.tight_layout()
  plt.show()
