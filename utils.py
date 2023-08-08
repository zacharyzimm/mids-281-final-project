import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from torchvision import transforms
import torch
from torch import nn
from PIL import Image

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

def detect_canny_edges(im, lower_bound=150, upper_bound=200):
    edges = cv2.Canny(im, lower_bound, (lower_bound+upper_bound)) #*.5)
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

def scale_value(X):
    original_min, original_max = (-1024, 3071)
    new_min, new_max = (0, 255)
    normalized_X = (X - original_min) / (original_max - original_min)
    scaled_X = normalized_X * (new_max - new_min) + new_min
    return scaled_X

def get_soft_tissue(image):
    soft_tissue_lower = scale_value(30)  # Choose the lower bound based on the soft tissue range
    soft_tissue_upper = scale_value(60)  # Choose the upper bound based on the soft tissue range

    # Apply windowing to filter the soft tissue in the CT scan
    windowed_image = np.clip(image, soft_tissue_lower, soft_tissue_upper)

    # Normalize the windowed image to the range [0, 1] for visualization
    windowed_image_normalized = (windowed_image - soft_tissue_lower) / (soft_tissue_upper - soft_tissue_lower)

    # # Display the original and windowed images
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original CT Scan')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.imshow(windowed_image_normalized, cmap='gray')
    # plt.title('Soft Tissue Filter')
    # plt.axis('off')

    masked_img = image.copy()
    masked_img[windowed_image_normalized==0] = 1

    # plt.subplot(1, 3, 3)
    # plt.imshow(masked_img, cmap='gray')
    # plt.title('Soft Tissue Filtered CT Scan')
    # plt.axis('off')

    # plt.suptitle(f'Class Label: {class_mappings[class_label]} ', fontsize=16)

    # plt.show()
    return masked_img

def get_resnet_features(img, model):
    
    model_conv_features = nn.Sequential(*list(model.children())[:-1]).to('cpu')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if np.max(img) > 1:
        img = img.astype(np.uint8)
    else:
        img = (img * 255.0).astype(np.uint8)
    img = Image.fromarray(img).convert('RGB')
    img = preprocess(img)

    return model_conv_features(img.unsqueeze(0).to('cpu')).squeeze().detach().numpy()

def transformer_feature_vector(image, model, extractor):
  # Preprocess the image using the feature extractor
  image = Image.fromarray(image).convert('RGB')
  inputs = extractor(images=image, return_tensors="pt")

  # Forward pass through the model's transformer (without the classification head)
  with torch.no_grad():
      feature_vector = model.vit(**inputs).last_hidden_state

  # Optionally, convert to a NumPy array
  feature_vector = feature_vector.numpy()

  return feature_vector

def extract_features(split_path, feature_func, kwargs=None):
    features = []
    images = []
    labels = []

    for label, class_name in class_mappings.items():
        class_path = os.path.join(split_path, class_name)
        for img_name in os.listdir(class_path):
            image = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            image = preprocess_image(image)
            if kwargs:
              feat = feature_func(image.copy(), **kwargs)
            else:
              feat = feature_func(image.copy())
            images.append(image)
            features.append(feat)
            labels.append(label)

    return images, features, labels


##### Dimmensionality Reduction

def get_PCA(X_list, n_components=2):
  pca_list = []
  xpca_list = []
  for X in X_list:
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X)
    X_pca = pca.transform(X)
    pca_list.append(pca)
    xpca_list.append(X_pca)
  return pca_list, xpca_list

def plot_PCA(X_list, labels, n_components=2):
  pca_list, xpca_list = get_PCA(X_list, n_components=n_components)

  plt.figure(figsize=(15,5))
  colors = ['b-', 'm-', 'r-']
  for i in range(len(X_list)):
    plt.plot(np.cumsum(pca_list[i].explained_variance_ratio_), colors[i], label=labels[i])
  plt.xticks(np.arange(n_components)+1)
  plt.yticks(np.linspace(0, 1, 8))
  plt.grid(True)
  plt.xlabel('Number of components')
  plt.ylabel('Explained Variances')
  plt.legend()
  plt.show()

def get_tsne(X_list, n_components=2):
  xtsne_list = []
  for X in X_list:
    tsne = TSNE(n_components=n_components, random_state=0)
    X_tsne = tsne.fit_transform(X)
    xtsne_list.append(X_tsne)
  return xtsne_list

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

def plot_image_histograms(img):
  # Calculate the histogram
  hist, bins = np.histogram(img.flatten(), 256, [0, 256])

  # Plot the histogram
  plt.figure(figsize=(8, 6))
  plt.bar(bins[:-1], hist, width=1, color='gray')
  plt.title("Grayscale Histogram")
  plt.xlabel("Pixel Value")
  plt.ylabel("Frequency")
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
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=4, min_samples_split=2, splitter='random')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

    return y_pred

def linear_svm(X_train, y_train, X_val, y_val=None):
  clf = SVC(C=0.1, gamma=0.1, kernel= 'linear')
  clf.fit(X_train, y_train)
  # X_df = pd.DataFrame({"x": X_train[:,0],
  #                     "y": X_train[:,1],
  #                     "label": y_train
  #                     })
  # ### plot linear SVM results
  # w = clf.coef_[0]
  # b = clf.intercept_

  # colors = np.random.rand(len(class_mappings.keys()), 3)

  # print(f"w = {w}")
  # print(f"b = {b}")

  # svmx = np.linspace(-2,2)
  # svmy = -w[0]/w[1]*svmx - b[0]/w[1]

  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # for l in class_mappings.keys():
  #     plt.scatter(X_df[X_df['label'] == l]['x'],
  #                 X_df[X_df['label'] == l]['y'],
  #                 c=colors[l], label=class_mappings[l])
  #     plt.plot(svmx, svmy, "m")
  #     plt.legend()
  #     plt.axis([-2, 2, -2, 2])
  # plt.show()

  y_pred = clf.predict(X_val)

  # accuracy = accuracy_score(y_val, y_pred)
  # print("Accuracy:", accuracy)

  return y_pred

def nonlinear_svm(X_train, y_train, X_val, y_val):
  clf = SVC(C=1e10, kernel="sigmoid")
  clf.fit(X_train, y_train)

  # Create a mesh grid to plot decision regions
  h = .02  # Step size in the mesh
  x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
  y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Obtain the predicted class labels for each point in the mesh grid
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  # Plot the decision regions
  plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

  # Plot the data points
  plt.scatter(X[:, 0], X[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')

  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.title('Nonlinear SVM Decision Regions')
  plt.show()

  y_pred = clf.predict(X_val)

  accuracy = accuracy_score(y_val, y_pred)
  print("Accuracy:", accuracy)