from flask import Flask, render_template, request, redirect, url_for, session, make_response
import requests
import os
import json
import tempfile
from skimage import io
from io import BytesIO
import numpy as np
import string
import random
import shutil
import pandas as pd
import pickle
import json
import glob
import time
import matplotlib
import matplotlib.animation as matplotlib_animation
matplotlib.use('Agg')
from helpers import dlmodel_ranking, get_features, visualize_decision, update_ranking

dt_model = pickle.load(open('models/quality_model_9-6.sav', 'rb'))

app = Flask(__name__)

def generate_id(length = 12, characters = string.ascii_lowercase + string.digits):
    return ''.join(random.choice(characters) for char in range(length))

@app.route('/<int:link_group>')
def home(link_group):
    link_group = str(link_group)
    if link_group == '99':
        link_group = '0'
    if 'user_id' in request.cookies:
        user_id = request.cookies.get('user_id')
    else:
        user_id = generate_id()
    if 'user_group' in request.cookies and link_group != '0':
        user_group = request.cookies.get('user_group')
    else:
        user_group = link_group
    resp = make_response(render_template('upload_images.html'))
    user_path = 'static/tmp/' + str(user_id)
    if os.path.isdir(user_path) == False:
        os.mkdir(user_path)
        empty_df = pd.DataFrame(columns = ['brightness', 'contrast', 'color',
        'areas_count', 'areas_0_size', 'areas_1_size', 'areas_2_size', 'sharpness1_area_0',
        'sharpness2_area_0', 'sharpness1_area_1', 'sharpness2_area_1', 'sharpness1_area_2',
        'sharpness2_area_2', 'sharpness1_crop_0', 'sharpness1_crop_1', 'noise'])
        empty_df.to_csv(user_path + '/feature_df.csv', index = True)
        dl_scores = []
        with open(user_path + '/dl_scores.json', 'w') as f:
            json.dump(dl_scores, f)

    current_millis = int(round(time.time()*1000))
    folders = glob.glob('static/tmp/*')
    for folder in folders:
        make_time = int(os.stat(folder).st_ctime) * 1000
        if (current_millis - make_time) > 5400000:
            shutil.rmtree(folder)

    resp.set_cookie('user_id', user_id)
    resp.set_cookie('user_group', user_group)
    return resp

@app.route('/file-upload', methods = ['GET', 'POST'])
def upload():
    user_id = request.cookies.get('user_id')
    user_path = 'static/tmp/' + str(user_id)
    if request.method == 'POST':
        uploaded = request.files.getlist('file')[0].read()
        image_file = io.imread(BytesIO(uploaded))
        image_name = generate_id(length = 6)
        io.imsave(user_path + '/' + image_name + '.jpg', image_file)

        features_animation = get_features(image_file, 224, animation = True)
        features_list = features_animation.get('features')
        feature_df = pd.read_csv(user_path + '/feature_df.csv', header = 0, index_col = 0)
        feature_df.loc[str(image_name)] = features_list
        feature_df.to_csv(user_path + '/feature_df.csv', index = True)

        animation = features_animation.get('visuals')
        animation.save(user_path + '/' + image_name + '.gif', writer = 'imagemagick')

        score_dl = dlmodel_ranking(user_path + '/' + image_name + '.jpg')
        with open(user_path + '/dl_scores.json', 'r') as f:
            dl_scores = json.load(f)
        dl_scores.append(score_dl)
        with open(user_path + '/dl_scores.json', 'w') as f:
            json.dump(dl_scores, f)

    return render_template('upload_images.html')

@app.route('/ranking', methods = ['GET', 'POST'])
def create_ranking():
    user_id = request.cookies.get('user_id')
    user_group = request.cookies.get('user_group')
    user_path = 'static/tmp/' + str(user_id)
    feature_df = pd.read_csv(user_path + '/feature_df.csv', header = 0, index_col = 0)
    images = []
    if user_group == '1':
        with open(user_path + '/dl_scores.json', 'r') as f:
            images = json.load(f)
            results = sorted(images, key = lambda x: x[1], reverse = True)
    else:
        for i, image in enumerate(feature_df.iterrows()):
            image_link = user_path + '/' + str(feature_df.index.values[i])
            feature_list = feature_df.iloc[i].values.tolist()
            consideration = visualize_decision(dt_model, feature_list)
            score = int(dt_model.predict([feature_list])[0])
            images.append([image_link, score, consideration])
        results = sorted(images, key = lambda x: x[1], reverse = True)
    return render_template('ranking_images.html', images = results)

@app.route('/dl-ranking', methods = ['GET', 'POST'])
def create_dl_ranking():
    user_id = request.cookies.get('user_id')
    user_path = 'static/tmp/' + str(user_id)
    with open(user_path + '/dl_scores.json', 'r') as f:
        images = json.load(f)
    results = sorted(images, key = lambda x: x[1], reverse = True)
    return render_template('update_ranking_grid.html', images = results)

@app.route('/new-ranking', methods = ['GET', 'POST'])
def create_new_ranking():
    selected_categories = json.loads(request.form["selected[]"])
    selected_categories.append('areas')
    user_id = request.cookies.get('user_id')
    user_group = request.cookies.get('user_group')
    user_path = 'static/tmp/' + str(user_id)
    images = []

    feature_df = pd.read_csv(user_path + '/feature_df.csv', header = 0, index_col = 0)
    dict_median_features = {'brightness': [], 'contrast': [], 'color': [], 'areas': [], 'sharpness': [], 'noise': []}
    median_df = feature_df.median(axis = 0)
    for item, value in median_df.iteritems():
        for key in dict_median_features.keys():
            if item.startswith(key):
                current_content = dict_median_features[key]
                current_content.append(value)
                dict_median_features[key] = current_content

    for i, image in enumerate(feature_df.iterrows()):
        feature_list = update_ranking(image, dict_median_features, selected_categories)
        image_link = user_path + '/' + str(feature_df.index.values[i])
        score = int(dt_model.predict([feature_list])[0])
        consideration = visualize_decision(dt_model, feature_list)
        images.append([image_link, score, consideration])

    results = sorted(images, key = lambda x: x[1], reverse = True)
    return render_template('update_ranking_grid.html', images = results)

if __name__ == '__main__':
    app.run()
