import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, data, img_as_float
from skimage.transform import resize



##### Preprocessing 

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

def crop_and_resize_images(split_path, class_mappings, out_img_size, out_img_dir):
    split = split_path.rpartition("/")[2]
    for label, class_name in class_mappings.items():
        class_path = os.path.join(split_path, class_name)
        if not os.path.exists(os.path.join(out_img_dir, split, class_name)):
            os.makedirs(os.path.join(out_img_dir, split, class_name))
        for img_name in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            cropped, _ = detect_border_and_crop(img)
            resized_img = resize(cropped, out_img_size, anti_aliasing=True)
            plt.imsave(os.path.join(out_img_dir, split, class_name, img_name), resized_img, cmap="gray")

            
def histogram_equalization(in_img):
    # compute cdf
    img_cdf, bins = exposure.cumulative_distribution(in_img, 256)
    # create empty array for all possible pixel values
    new_cdf = np.zeros(256)
    # populate array with values from cdf
    # use bins as the index into the array
    new_cdf[bins] = img_cdf
    # create empty array the same size as the image
    out_img = np.zeros(in_img.shape, dtype=in_img.dtype)
    # for each pixel, look up the value from the cdf
    for i in range(out_img.shape[0]):
        for j in range(out_img.shape[1]):
            out_img[i, j] = (new_cdf[ in_img[i, j] ] * 255)

    return out_img

def equalize_images(split_path, class_mappings, out_img_dir):
    split = split_path.rpartition("/")[2]
    for label, class_name in class_mappings.items():
        class_path = os.path.join(split_path, class_name)
        if not os.path.exists(os.path.join(out_img_dir, split, class_name)):
            os.makedirs(os.path.join(out_img_dir, split, class_name))
        for img_name in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            equalized_img = histogram_equalization(img)
            plt.imsave(os.path.join(out_img_dir, split, class_name, img_name), equalized_img, cmap="gray")

            

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

def extract_features(split_path, feature_func, class_mappings):
    features = []
    images = []
    labels = []

    for label, class_name in class_mappings.items():
        class_path = os.path.join(split_path, class_name)
        for img_name in os.listdir(class_path):
            image = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            feat = feature_func(image)
            images.append(image)
            features.append(feat)
            labels.append(label)

    return images, features, labels

def scale_value(X):
    original_min, original_max = (-1024, 3071)
    new_min, new_max = (0, 255)
    normalized_X = (X - original_min) / (original_max - original_min)
    scaled_X = normalized_X * (new_max - new_min) + new_min
    return scaled_X

def get_soft_tissue(image):
    soft_tissue_lower = scale_value(30)  # Choose the lower bound based on the soft tissue range for Hounsfield units
    soft_tissue_upper = scale_value(60)  # Choose the upper bound based on the soft tissue range for Hounsfield units

    # Apply windowing to filter the soft tissue in the CT scan
    windowed_image = np.clip(image, soft_tissue_lower, soft_tissue_upper)

    windowed_image_normalized = (windowed_image - soft_tissue_lower) / (soft_tissue_upper - soft_tissue_lower)

    masked_img = image.copy()
    masked_img[windowed_image_normalized==0] = 1

    return masked_img


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
