import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class_mappings = {
    0: "normal",
    1: "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    2: "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    3: "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"
}

##### Preprocessing 

def preprocess_image(image, crop_and_resize=False, out_img_size=(256, 256)):
  if crop_and_resize:
    ## Crop image
    image, _ = detect_border_and_crop(image)
    ## Resize image
    image = resize(image, out_img_size, anti_aliasing=True)

  ## Apply histogram equalization to the image
  image = cv2.equalizeHist(image)

  return image

def detect_border_and_crop(image):
  kernel_size = 5
  sigma = 1.0
  kernel = cv2.getGaussianKernel(kernel_size, sigma)
  kernel = np.outer(kernel, kernel.transpose())

  # Apply the Gaussian filter
  gray = cv2.filter2D(image, -1, kernel)

  # threshold to get just the signature
  _, thresh_gray = cv2.threshold(gray, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

  # find where the signature is and make a cropped region
  points = np.argwhere(thresh_gray!=0) # find where the black pixels are
  points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
  
  try:
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image
  except:
    crop = gray
  
  # get the thresholded crop
  _, thresh_crop = cv2.threshold(crop, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

  return crop, thresh_crop

def crop_and_resize_images(split_path, out_img_size, out_img_dir):
  split = split_path.rpartition("/")[2]
  for label, class_name in class_mappings.items():
    class_path = os.path.join(split_path, class_name)
    if not os.path.exists(os.path.join(out_img_dir, split, class_name)):
        os.makedirs(os.path.join(out_img_dir, split, class_name))
    for img_name in os.listdir(class_path):
      img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
      cropped, _ = detect_border_and_crop(img)
      resized_img = resize(img, out_img_size, anti_aliasing=True)
      plt.imsave(os.path.join(out_img_dir, split, class_name, img_name), resized_img, cmap="gray")


##### Feature extraction

def get_average_image_size(img_dir):
  img_sizes = []

  for img_name in os.listdir(img_dir):
      try:
          img = plt.imread(img_dir + img_name)
          img_sizes.append([img.shape[0], img.shape[1]])
      except:
          print(img_name + " corrupt")

  img_sizes = np.array(img_sizes)

  print("Average Image Sizes")
  m_mean = np.mean(img_sizes[:, 0])
  n_mean = np.mean(img_sizes[:, 1])
  print(f"    mean:   ({m_mean}, {n_mean})")

  m_median = np.median(img_sizes[:, 0])
  n_median = np.median(img_sizes[:, 1])
  print(f"    median: ({m_median}, {n_median})")

  return [m_mean, n_mean], [m_median, n_median]

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

def threshold_image(img, thresh=100):
  img[img > thresh] = 0
  return img

def extract_features(split_path, feature_func):
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

def plot_features(imgs, features, labels, idx, feature_name):
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


##### Classifiers

def cosine_similarity_kmeans(train_features, num_clusters=4):
  features = np.array([img.flatten() for img in train_features])

  # Calculate cosine similarity matrix
  cosine_sim_matrix = cosine_similarity(features)

  # Create the KMeans model with cosine similarity
  kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
  kmeans.fit(cosine_sim_matrix)

  # Get cluster labels for each data point
  preds = kmeans.labels_

  # Get cluster centers (representative points)
  cluster_centers = kmeans.cluster_centers_

  return preds, cluster_centers

def plot_confusion_matrix(true_labels, pred_labels):
  conf_matrix = confusion_matrix(true_labels, pred_labels)
  classes = [x[1].partition("_")[0] for x in class_mappings.items()]
  sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt="d", xticklabels=classes, yticklabels=classes)

def classify_decision_tree(X_train, y_train, X_val, y_val):
    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

    return y_pred