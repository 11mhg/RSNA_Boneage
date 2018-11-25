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

def get_dataframe():
    data_dir = './rsna-bone-age/'
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

    train_df, val_df = train_test_split(data,
                                test_size = 0.25,
                                random_state=7,
                                stratify=data['boneage_category'])
    return train_df, val_df, bone_age_div


def open_image(filepath,dim=(384,384)):
    image = Image.open(filepath)
    image = image.resize(dim,resample=Image.BILINEAR)
    return np.array(image)

def resize(image,size):
    return cv2.resize(image,size)

def normalize(image):
    image = np.array(image,dtype=np.float32)
    if image.max() > 1.0:
        image = np.interp(image,(image.min(),image.max()),(0.0,1.0))
    image = np.array(image,dtype=np.float32)
    return image

def augment_image(image):
    if not hasattr(augment_image,'pipe'):
        sometimes = lambda aug: iaa.Sometimes(0.5,aug)
        augment_image.pipe = iaa.Sequential(
                [
                    iaa.SomeOf((0,6),
                        [
                            iaa.OneOf([
                                iaa.GaussianBlur((0,3.0)),
                                iaa.AverageBlur(k=(2,7)),
                                iaa.MedianBlur(k=(3,11)),
                            ]),
                        iaa.Sharpen(alpha=(0,1.0),lightness=(0.75,1.5)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,0.05*255),per_channel=0.5),
                        iaa.Dropout((0.01,0.075),per_channel=0.5),
                        iaa.Add((-5,5),per_channel=0.5),
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

def get_data(df):
    labels = []
    img_paths = []
    pbar = tqdm(df.iterrows())
    for i, row in pbar:
        labels.append(row['bone_age_zscore'])
        img_paths.append(row['path'])
    return labels, img_paths


def custom_mae_metric(y_true, y_pred):
    global bone_age_div 
    return keras.metrics.mean_absolute_error(bone_age_div*y_true,bone_age_div*y_pred)



class RSNAGenerator(keras.utils.Sequence):
    'Data Generator for Keras'

    def __init__(self,batch_size=32,dim=(384,384),train=True,shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()
        self.prep()

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self,index):
        ind = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        cur_labels, cur_images = [self.labels[k] for k in ind], [self.images[k] for k in ind]
        cur_images = [resize(img,self.dim) if img.shape[:2] != self.dim else img for img in cur_images]
        X = np.empty((self.batch_size, *self.dim,1),dtype=np.float32)
        Y = np.empty((self.batch_size),dtype=np.float32)
        for i in range(self.batch_size):
            X[i,] = np.expand_dims(cur_images[i],-1)
            Y[i] = cur_labels[i]
        return X,Y

    def prep(self):
        self.images=[]
        pbar = tqdm(range(len(self.labels)))
        pbar.set_description("Loading dataset into memory for quick training")
        for ind in pbar:
            if self.train:
                self.images.append(augment_image(open_image(self.img_paths[ind],self.dim)))
            else:
                self.images.append(open_image(self.img_paths[ind],self.dim))
    
    def on_epoch_end(self):
        global bone_age_div
        train_df, val_df, bone_age_div = get_dataframe()
        if self.train:
            self.labels, self.img_paths = get_data(train_df)
        else:
            self.labels, self.img_paths = get_data(val_df)
        
        self.indexes = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indexes)
