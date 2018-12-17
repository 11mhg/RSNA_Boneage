import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random
import os,sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa
import keras
import cv2
import time
from multiprocessing import Pool

'''
Functions and class for utility
Includes image opening, dataframe retrieval,
image augmentation, normalization and custom mae metric
'''

#Retrieves the train, val dataframes and the bone age div and bone age mean from the dataset
def get_dataframe(directory):
    data_dir = directory + '/rsna-bone-age/'
    data = pd.read_csv(os.path.join(data_dir,'boneage-training-dataset.csv'))
    data['path'] = data['id'].map(lambda x: os.path.join(data_dir,'boneage-training-dataset',
        '{}.png'.format(x)))
    data['exists'] = data['path'].map(os.path.exists)
    data['gender'] = data['male'].map(lambda x: 'male' if x else 'female')
    data['female'] = data['male'].map(lambda x: False if x else True)

    bone_age_mean = data['boneage'].mean()
    bone_age_div = 2*data['boneage'].std()
    data['bone_age_zscore'] = data.boneage.map(lambda x: (x-bone_age_mean)/bone_age_div)
    data.dropna(inplace=True)

    data['boneage_category'] = pd.cut(data['boneage'],10)

    train_df, test_df = train_test_split(data,
                                test_size = 0.15,
                                random_state=7,
                                stratify=data['boneage_category'])
    train_df, val_df = train_test_split(train_df,test_size=0.25,
                                random_state=7,
                                stratify=train_df['boneage_category'])
    return train_df, val_df, bone_age_div, bone_age_mean

#Retrieves only the test set 
#So long as random state is the same, this test set is the same as the above test set
def get_test(directory):
    data_dir = directory+'/rsna-bone-age/'
    data = pd.read_csv(os.path.join(data_dir,'boneage-training-dataset.csv'))
    data['path'] = data['id'].map(lambda x: os.path.join(data_dir, 'boneage-training-dataset','{}.png'.format(x)))
    data['exists'] = data['path'].map(os.path.exists)
    data['gender'] = data['male'].map(lambda x: 'male' if x else 'female')
    data['female'] = data['male'].map(lambda x: False if x else True)
    data['boneage_category'] = pd.cut(data['boneage'],10)
    _, test_df = train_test_split(data,
                test_size=0.15,
                random_state=7,
                stratify=data['boneage_category'])
    return test_df

#A wrapper function to allow for argument unpacking
def worker_unpack(args):
    return open_images_worker(*args)

#A wrapper function for use with threads
def open_images_worker(imgname,dim=(384,384),train=True):
    if train:
        return augment_image(open_image(imgname,dim))
    else:
        return open_image(imgname,dim)

#Opens an image and resizes if need be
def open_image(filepath,dim=(384,384)):
    image = Image.open(filepath)
    image = resize(np.array(image),dim)
    return image

#Helper function for resizing
def resize_unpack(args):
    return resize(*args)

#Resize function using cv2
def resize(image,size):
    return cv2.resize(image,size)

#Normalization of image
def normalize(image):
    image = np.array(image,dtype=np.float32)
    if image.max() > 1.0:
        image = image/255.0 
    image = np.array(image,dtype=np.float32)
    return image

#Augments image and if it has already been called once
#it will save the augment pipe so that it can quickly reaugment based on the same
#pipe
def augment_image(image):
    if not hasattr(augment_image,'pipe'):
        sometimes = lambda aug: iaa.Sometimes(0.5,aug)
        augment_image.pipe = iaa.Sequential(
                [
                    iaa.SomeOf((0,6),
                        [
                        iaa.Sharpen(alpha=(0,1.0),lightness=(0.75,1.5)),
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.Affine(rotate=(-90,90),
                            translate_percent={"x":(-0.15,0.15),"y":(-0.15,0.15)},
                        ),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
    image = augment_image.pipe.augment_image(np.array(image,dtype=np.uint8))
    image = normalize(image)
    return image

#Retrieves the labels, paths and genders for a particular dataframe
def get_data(df):
    labels = []
    img_paths = []
    genders = []
    pbar = tqdm(df.iterrows())
    for i, row in pbar:
        labels.append(row['bone_age_zscore'])
        img_paths.append(row['path'])
        genders.append(1.0 if row['female'] else 0.0)
    return labels, img_paths, genders

#custom metric for mean average error (requires the global bone_age_div in order to perform the mae calculation)
def custom_mae_metric(y_true, y_pred):
    global bone_age_div 
    return keras.metrics.mean_absolute_error(bone_age_div*y_true,bone_age_div*y_pred)


#This class is a data genereator for keras
class RSNAGenerator(keras.utils.Sequence):
    'Data Generator for Keras'

    def __init__(self,directory,batch_size=32,dim=(384,384),train=True,shuffle=True):
        self.directory = directory
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.on_init()
        self.on_epoch_end()
        self.prep()

#Returns the length of the dataset
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

#Returns a single batch at specified index
    def __getitem__(self,index):
        ind = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        cur_labels, cur_images = [self.labels[k] for k in ind], [self.images[k] for k in ind]
        cur_genders = [self.genders[k] for k in ind]
        if cur_images[0].shape[:2] != self.dim:
            p = Pool(processes=20)
            args = []
            for i in range(len(self.images)):
                args.append((self.images[i],self.dim))
            self.images = p.map(resize_unpack,args)
            p.terminate()
        X = np.empty((self.batch_size, *self.dim,1),dtype=np.float32)
        Y = np.empty((self.batch_size),dtype=np.float32)
        gends = np.empty((self.batch_size,1),dtype=np.float32)
        for i in range(self.batch_size):
            X[i,] = np.expand_dims(cur_images[i],-1)
            Y[i] = cur_labels[i]
            gends[i,0] = cur_genders[i]
        return [X,gends],Y

#Returns all elements in the dataset
    def __getall__(self):
        cur_labels = np.array(self.labels,dtype=np.float32)
        cur_images = np.array(self.images,dtype=np.float32)
        cur_labels = np.expand_dims(cur_labels,-1)
        cur_images = np.expand_dims(cur_images,-1)
        return cur_images,cur_labels

#Preps the generator by loading images into memory and augmenting them if need be
    def prep(self):
        self.images=[]
        print("Loading dataset into memory for faster training")
        p = Pool(processes=20)
        start = time.time()
        filenames = self.img_paths
        args = []
        for i in range(len(filenames)):
            args.append((filenames[i],self.dim,self.train))
        self.images = p.map(worker_unpack,args)
        p.terminate()
        print("Done prepping")

#Initialize dataframe and retrieve appropriate values
    def on_init(self):
        global bone_age_div
        self.bad = 0
        train_df, val_df, bone_age_div, bone_age_mean = get_dataframe(self.directory)
        if self.train:
            self.labels, self.img_paths,self.genders = get_data(train_df)
        else:
            self.labels, self.img_paths,self.genders = get_data(val_df)
        self.bad = bone_age_div
        self.mean = bone_age_mean

#When the epeoch ends, we reshuffle indices
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indexes)
