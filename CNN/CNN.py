# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:29:59 2019

@author: 41410
"""
import numpy as np
import cv2
from glob import glob
import os
from sklearn.model_selection import train_test_split
import tflearn
import xml.etree.ElementTree as ET
from PIL import Image

make_dataset = True
train = False
load = True
test = True

IMG_WIDTH, IMG_HEIGHT = 32, 32
NB_EPOCHS = 20
BATCH_SIZE = 32
n_classes = 120

if make_dataset == True:
    breed_list = os.listdir("Images/")
    num_classes = len(breed_list)
    n_total_images = 0
    for breed in breed_list:
        n_total_images += len(os.listdir("Images/{}".format(breed)))

    label_maps = {}
    label_maps_rev = {}
    for i, v in enumerate(breed_list):
        label_maps.update({v: i})
        label_maps_rev.update({i : v})

    if not os.path.exists('New'):
        os.mkdir('New')
        for breed in breed_list:
            os.mkdir('New/' + breed)
        print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('New'))))

        for breed in os.listdir('New'):
            for file in os.listdir('Annotation/{}'.format(breed)):
                img = Image.open('Images/{}/{}.jpg'.format(breed, file))
                tree = ET.parse('Annotation/{}/{}'.format(breed, file))
                xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
                xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
                ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
                ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
                img = img.crop((xmin, ymin, xmax, ymax))
                img = img.convert('RGB')
                img = img.resize((32,32))
                img.save('New/' + breed + '/' + file + '.jpg')
    else:
        print("You have already cropped the dataset!")

def convert(img):
    info = np.iinfo(img.dtype)
    return img.astype(np.float32)/info.max

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
    return convert(img)

def onehot(y, n_classes):
    tmp = np.zeros((y.shape[0], n_classes))
    tmp[np.arange(y.shape[0]), y.astype(int)] = 1
    return np.array(tmp, dtype=np.float32)


image_paths = glob('New/*')
breeds = [s[14:] for s in image_paths]

n = 5
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.0)

net = tflearn.input_data(shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(net, n_classes, activation='softmax') # Multi class classification basically with softmax


opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt,
                             loss='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path='model_resnext_dog_breeds',
                        max_checkpoints=10, tensorboard_verbose=3,
                        clip_gradients=0.)

if train == True:
    X = []
    y = []
    for idx, path in enumerate(image_paths):
        for image in glob(path + "/*"):
            X.append(load_img(image))
            y.append(idx)
        print(idx, 'Loaded', breeds[idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print('Data loaded!')
    y = onehot(y, n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89741)
    model.fit(X_train, y_train, n_epoch=NB_EPOCHS, validation_set=(X_test, y_test),
              snapshot_epoch=False, snapshot_step=500,
              show_metric=True, batch_size=BATCH_SIZE, shuffle=True,
              run_id='try')
    model.save("Best-model")

if load == True:
    if os.path.isfile('{}.meta'.format('Best-model')):
        model.load('Best-model')
        print('model loaded!')
    else:
        print('Error loading model!')

if test == True:
    test_file = input("Enter image path to classify: ")
    while test_file!='exit':
        test_file = test_file.replace("'", "")
        test_file = test_file.replace(" ", "")
        testimg = load_img(test_file)
        pred = model.predict([testimg])[0]
        ans = pred.argsort()[-3:][::-1]
        print("That's most likely a", breeds[ans[0]])
        print("But it might also be a", breeds[ans[1]], "or a", breeds[ans[2]])
        test_file = input("Enter image path to classify: ")