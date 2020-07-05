import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from skimage import io
from skimage.transform import resize

import numpy as np
import pandas as pd

from skimage import io, color, exposure, filters, measure
from skimage.transform import resize

from PIL import Image

from scipy import stats
import scipy.ndimage.filters

from sklearn import cluster
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as matplotlib_animation

import warnings
warnings.filterwarnings('ignore')


from kutils import applications as apps
from kutils import image_utils as iu

import tensorflow as tf

from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
from keras import backend as K

import logging
logging.getLogger('tensorflow').disabled = True

# build scoring model
base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
head = apps.fc_layers(base_model.output, name='fc',
                      fc_sizes      = [2048, 1024, 256, 1],
                      dropout_rates = [0.25, 0.25, 0.5, 0],
                      batch_norm    = 2)

model = Model(inputs = base_model.input, outputs = head)

# load pretrained model weights
model.load_weights('models/bsz64_i1[224,224,3]_lMSE_o1[1]_best_weights.h5')
graph = tf.get_default_graph()

def dlmodel_ranking(name):

    # ims_candidates = []
    # images = glob.glob('static/tmp/' + user_id + '/*' + '.jpg')
    category_scores = {'brightness': 0, 'contrast': 0, 'color': 0, 'areas': 0, 'sharpness': 0, 'noise': 0}

    # for name in images:
        # im = io.imread(name)
        # rescaled_im = resize(im, (224, 224), anti_aliasing = True)
    im = Image.open(name)
    rescaled_image = ImageOps.fit(im, (224, 224), Image.ANTIALIAS)
    # ims_candidates.append([rescaled_im, name])

    predictions = []
    # for im in ims_candidates:
    im_preprocessed = preprocess_fn(image.img_to_array(rescaled_image))

    # Create a batch to feed to the model
    batch = np.expand_dims(im_preprocessed, 0)

    # Predict the image's quality score
    with graph.as_default():
        y_pred = model.predict(batch)

    # predictions.append([name.split('.jpg')[0], int(y_pred[0][0]), [int(y_pred[0][0]), category_scores]])

    # return(predictions)
    return(name.split('.jpg')[0], int(y_pred[0][0]), [int(y_pred[0][0]), category_scores])

def get_features(full_img, rescale_factor, animation):

    # Image preprocessing: rescaling and clustering
    rescaled_img, rescaled_img_gray, mode = preprocess(full_img, rescale_factor)
    rescaled_img_luv, kmeans_cluster, rescaled_img_range = segmentation(rescaled_img, rescaled_img_gray, mode, True)

    # Calculate features for rescaled image and clusters
    exposure, contrast = feature_exposure(rescaled_img_gray)
    rescaled_img_hsv, saturation = feature_saturation(rescaled_img, mode)
    var_0, max_0, len_0, var_1, max_1, len_1, var_2, max_2, len_2, number_areas, rescaled_laplacian = feature_segments(rescaled_img, kmeans_cluster, rescaled_img_range)

    # Calculate cropped image by salience
    center_y, center_x, salience_img, salience_img_th = calculate_salience(rescaled_img_gray, rescaled_img_range, rescaled_img_hsv, rescale_factor, mode)
    cropped_img, cropped_img_gray = salience_crop(center_x, center_y, full_img, rescale_factor, mode)

    # Calculate features for cropped image (more detail)
    cropped_img_luv, cropped_kmeans_cluster, cropped_img_range = segmentation(cropped_img, cropped_img_gray, mode, False)
    var_crop_0, var_crop_1, noise_crop = feature_crop(cropped_img, cropped_kmeans_cluster, cropped_img_range)

    # load gif for animation of process
    if animation == True:
        fig = plt.figure(frameon = False, dpi = 100)
        fig.set_size_inches(2.24, 2.24)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        animation_ims = []
        image_list = [rescaled_img, rescaled_img, kmeans_cluster, rescaled_laplacian, salience_img, salience_img_th, rescaled_img, cropped_img, cropped_img]
        for i, image in enumerate(image_list):
            frame = ax.imshow(image, cmap = 'gray', aspect = 'auto')
            animation_ims.append([frame])
            if 5 <= i <= 6:
                dot = ax.plot(center_x * rescale_factor, center_y * rescale_factor, marker = 'o', markerfacecolor = '#13e27a', markeredgecolor = '#13e27a')
                animation_ims[i].append(dot[0])
        process_animation = matplotlib_animation.ArtistAnimation(fig, animation_ims, interval=2000, blit=True, repeat_delay=0)
    return({'features': [exposure, contrast, saturation, number_areas, len_0, len_1, len_2, var_0, max_0, var_1, max_1, var_2, max_2, var_crop_0, var_crop_1, noise_crop], 'visuals': process_animation})

def features_to_df(feature_list):
    df = pd.DataFrame([feature_list])
    df.columns = ['brightness', 'contrast', 'color', 'areas_count', 'areas_0_size', 'areas_1_size', 'areas_2_size', 'sharpness1_area_0', 'sharpness2_area_0', 'sharpness1_area_1', 'sharpness2_area_1', 'sharpness1_area_2', 'sharpness2_area_2', 'sharpness1_crop_0', 'sharpness1_crop_1', 'noise']
    return(df)

def features_from_df(df, selected_categories):
    dict_features = {'brightness': [], 'contrast': [], 'color': [], 'areas': [], 'sharpness': [], 'noise': []}
    for item, value in df.iteritems():
        for key in dict_features.keys():
            if item.startswith(key):
                current_content = dict_features[key]
                current_content.append(value[0])
                dict_features[key] = current_content

    all_categories = ['brightness', 'contrast', 'color', 'areas', 'sharpness', 'noise']
    feature_list = []
    for category in all_categories:
        features_in_category = dict_features.get(category)
        if category in selected_categories:
            for feature in features_in_category:
                feature_list.append(feature)
        else:
            for feature in features_in_category:
                feature_list.append(0)
    return(feature_list)

def preprocess(full_img, rescale_factor):
    rescaled_img = resize(full_img, (rescale_factor, rescale_factor), anti_aliasing = True)
    if rescaled_img.shape != (rescale_factor, rescale_factor, 3):
        mode = 'grayscale'
        rescaled_img_grayscale = rescaled_img
    elif np.array_equal(rescaled_img[:, :, 0], rescaled_img[:, :, 1]) and np.array_equal(rescaled_img[:, :, 0], rescaled_img[:, :, 2]):
        mode = 'grayscale'
        rescaled_img_gray = color.rgb2gray(rescaled_img)
    else:
        mode = 'color'
        rescaled_img_gray = color.rgb2gray(rescaled_img)
    return(rescaled_img, rescaled_img_gray, mode)

def segmentation(img, img_gray, mode, calculate_number_clusters):
    img_range = (img_gray - np.amin(img_gray)) * 255.0 / (np.amax(img_gray) - np.amin(img_gray))

    if mode == 'color':
        img_luv = color.rgb2luv(img)
    else:
        img_luv = np.stack((img_range, img_range, img_range), axis = 2)

    img_med = scipy.ndimage.median_filter(img_luv, size = 2)

    x, y = img.shape[:2]
    img_reshape = img_med.reshape(x*y, 3)

    if calculate_number_clusters == True:
        distortions = []
        for i in range(1, 4):
            kmeans = cluster.MiniBatchKMeans(max_iter = 2, batch_size = 2000, n_clusters = i)
            kmeans.fit(img_reshape)
            distortions.append(kmeans.inertia_)

        coef = []
        change_coef = []

        for i in range(1, 3):
            coef.append((distortions[i - 1] / distortions[i]))

        for i in range(1, 2):
            change_coef.append(coef[i] / coef[i - 1])

        count_k = 2
        for number in change_coef[1:]:
            if number < 0.8:
                count_k += 1
            else:
                break
    else:
        count_k = 2

    kmeans = cluster.MiniBatchKMeans(max_iter = 10, batch_size = 2000, n_clusters = count_k)
    kmeans.fit(img_reshape)
    centers = kmeans.cluster_centers_
    labels = (kmeans.labels_)
    clustered = centers[labels].reshape(x, y, 3)
    clustered_int = color.rgb2gray(clustered.astype(int))

    return(img_luv, clustered_int, img_range)

def calculate_salience(img_gray, img_range, img_hsv, rescale_factor, mode):
#     https://jacobgil.github.io/computervision/saliency-from-backproj
    if mode == 'grayscale':
        black_image = np.zeros((rescale_factor, rescale_factor, 3))
        return(0.5, 0.5, black_image, black_image)

    ravel_sat = np.ravel(img_hsv[:, :, 1])
    ravel_hue = np.ravel(img_hsv[:, :, 0])

    full_hist, a, b = np.histogram2d(ravel_sat, ravel_hue, bins = ([0, 0.5, 1]))

    sorted_values = sorted(full_hist.ravel())

    new_image = []
    for i, pixel in enumerate(ravel_sat):
        if pixel < 0.5:
            if ravel_hue[i] < 0.5:
                new_pixel = 240 - sorted_values.index(full_hist[0][0]) * 80
            else:
                new_pixel = 240 - sorted_values.index(full_hist[0][1]) * 80
        else:
            if ravel_hue[i] < 0.5:
                new_pixel = 240 - sorted_values.index(full_hist[1][0]) * 80
            else:
                new_pixel = 240 - sorted_values.index(full_hist[1][1]) * 80
        new_image.append(new_pixel)

    salience_image = np.array(new_image).reshape((rescale_factor, rescale_factor))
    salience_image = scipy.ndimage.gaussian_filter(salience_image, sigma = 8)
    salience_image = exposure.equalize_hist(salience_image)

    threshold = filters.threshold_otsu(salience_image)
    salience_image_th = (salience_image > threshold).astype(int)

    properties = measure.regionprops(salience_image_th, img_gray)
    weighted_center_of_mass = properties[0].centroid

    return(weighted_center_of_mass[0]/rescale_factor, weighted_center_of_mass[1]/rescale_factor, salience_image, salience_image_th)

def salience_crop(center_x, center_y, full_img, rescale_factor, mode):
    height, width = full_img.shape[:2]
    x = int(center_x * width)
    y = int(center_y * height)
    original_x, original_y = x, y

    half_crop = 0.5 * rescale_factor

    if x < half_crop:
        x = half_crop
    if y < half_crop:
        y = half_crop

    if (width - x) < half_crop:
        x = width - half_crop
    if (height - y) < half_crop:
        y = height - half_crop

    x_min = int(x - half_crop)
    x_max = int(x + half_crop)
    y_min = int(y - half_crop)
    y_max = int(y + half_crop)

    img_crop = full_img[y_min:y_max, x_min:x_max]
    if mode == 'grayscale':
        img_crop_gray = img_crop
    else:
        img_crop_gray = color.rgb2gray(img_crop)

    return(img_crop, img_crop_gray)

def feature_exposure(img):
    return((img.ravel()).var(), img.std())

def feature_saturation(img, mode):
    img_hsv = np.zeros((img.shape[0], img.shape[1], 3))
    if mode == 'color':
        img_hsv = color.rgb2hsv(img)
    saturation = img_hsv[:, :, 1]
    mean_saturation = np.array(saturation).mean()
    return(img_hsv, mean_saturation)

def feature_segments(img, kmeans_cluster, img_range):
    seg_ravel = kmeans_cluster.ravel()

    kernel = np.ones((3, 3))
    kernel[1,1] = -8
    laplacian = scipy.ndimage.filters.convolve(img_range, kernel)
    ravel_laplacian = laplacian.ravel()

    ravel_img = img.ravel()

    areas = []
    for item in np.unique(seg_ravel):
        list_laplacian = []
        list_img = []
        for i, pixel in enumerate(seg_ravel):
            if pixel == item:
                list_laplacian.append(ravel_laplacian[i])
                list_img.append(ravel_img[i])
        array_img = np.array(list_img)
        array_laplacian = np.array(list_laplacian)
        areas.append([array_laplacian.var(), array_laplacian.max(), len(array_img)/len(seg_ravel)])

    number_clusters = len(areas)

    for i in range(number_clusters, 3):
        areas.append([0, 0, 0])

    def key(item):
        return(item[2])

    sorted_areas = sorted(areas, key = key)

    return(sorted_areas[0][0], sorted_areas[0][1], sorted_areas[0][2],
            sorted_areas[1][0], sorted_areas[1][1], sorted_areas[1][2],
            sorted_areas[2][0], sorted_areas[2][1], sorted_areas[2][2],
            number_clusters, laplacian)

def feature_crop(img, kmeans_cluster, img_range):
    seg_ravel = kmeans_cluster.ravel()

    kernel = np.ones((3, 3))
    kernel[1,1] = -8
    laplacian = scipy.ndimage.filters.convolve(img_range, kernel)
    ravel_laplacian = laplacian.ravel()

    areas = []
    for item in np.unique(seg_ravel):
        list_laplacian = []
        for i, pixel in enumerate(seg_ravel):
            if pixel == item:
                list_laplacian.append(ravel_laplacian[i])
        array_laplacian = np.array(list_laplacian)
        areas.append(array_laplacian.var())
    sorted_areas = sorted(areas)

    full_laplacian = sorted(ravel_laplacian)
    noise_correction = int(len(full_laplacian) * 0.98)
    corrected_laplacian = full_laplacian[:noise_correction]
    noise = np.array(corrected_laplacian).max() / np.array(full_laplacian).max()

    return(sorted_areas[0], sorted_areas[1], noise)

def calculate_similarity(img1, img2):
    similarity = np.sum((img1.astype(float) - img2.astype(float)) ** 2)
    similarity = similarity / float(img1.shape[0] * img1.shape[1])
    return(similarity)

def visualize_decision(model, sample):
    feature = model.tree_.feature
    value = model.tree_.value
    node_indicator = model.decision_path([sample])
    feature_names = ['brightness', 'contrast', 'color', 'areas_count', 'areas_0_size', 'areas_1_size', 'areas_2_size', 'sharpness1_area_0', 'sharpness2_area_0', 'sharpness1_area_1', 'sharpness2_area_1', 'sharpness1_area_2', 'sharpness2_area_2', 'sharpness1_crop_0', 'sharpness1_crop_1', 'noise']
    category_scores = {'brightness': 0, 'contrast': 0, 'color': 0, 'areas': 0, 'sharpness': 0, 'noise': 0}
    node_index = node_indicator.indices[node_indicator.indptr[0]:
                                        node_indicator.indptr[1]]

    base_score = previous_score = value[0][0][0]
    for i, node_id in enumerate(node_index[:-1]):
        current_score = value[node_index[i + 1]][0][0]
        difference = current_score - previous_score
        previous_score = current_score
        feature_name = feature_names[feature[node_id]]

        for key in category_scores.keys():
            if feature_name.startswith(key):
                previous_cumulative = category_scores[key]
                current_cumulative = previous_cumulative + difference
                category_scores[key] = current_cumulative

        for key in category_scores:
            category_scores[key] = round(category_scores[key], 1)

    return([round(base_score, 1), category_scores])

def update_ranking(image, dict_median_features, selected_categories):
    dict_features = {'brightness': [], 'contrast': [], 'color': [], 'areas': [], 'sharpness': [], 'noise': []}
    for item, value in image[1].iteritems():
        for key in dict_features.keys():
            if item.startswith(key):
                current_content = dict_features[key]
                current_content.append(value)
                dict_features[key] = current_content
    all_categories = ['brightness', 'contrast', 'color', 'areas', 'sharpness', 'noise']
    feature_list = []
    for category in all_categories:
        if category in selected_categories:
            features_in_category = dict_features.get(category)
            for feature in features_in_category:
                feature_list.append(feature)
        else:
            features_in_category = dict_median_features.get(category)
            for feature in features_in_category:
                feature_list.append(feature)
    return(feature_list)
