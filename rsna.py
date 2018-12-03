import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random

import os
import matplotlib.pyplot as plt
from skimage.io import imread

data_dir = './rsna-bone-age/'
data = pd.read_csv(os.path.join(data_dir,'boneage-training-dataset.csv'))

data['path'] = data['id'].map(lambda x: os.path.join(data_dir,'boneage-training-dataset',
                                                     '{}.png'.format(x)))
data['exists'] = data['path'].map(os.path.exists)

print(data['exists'].sum(), ' images found of', data.shape[0], 'total')

data['gender'] = data['male'].map(lambda x: 'male' if x else 'female')
data['female'] = data['male'].map(lambda x: False if x else True)
print(data.gender.value_counts())


bone_age_mean = data['boneage'].mean()
bone_age_std = data['boneage'].std()
print(bone_age_mean,' is the mean of the boneages')
print(bone_age_std, ' is the standard deviation of the boneages')
bone_age_div = 2*data['boneage'].std()

data['bone_age_zscore'] = data.boneage.map(lambda x: (x-bone_age_mean)/bone_age_div)
data['bone_age_truezscore'] = data.boneage.map(lambda x: (x-bone_age_mean)/bone_age_std)

a = data.hist(column='boneage')

plt.title('Distirbution of BoneAge')
plt.savefig('./plots/bone_age_hist.jpg')
b = data.hist(column='bone_age_zscore')
plt.title('Distribution of Zn-score')
plt.savefig('./plots/bone_age_2zscore_hist.jpg')
c= data.hist(column='bone_age_truezscore')
plt.title('Distribution of true Z-Score')
plt.savefig('./plots/bone_age_truezscore.jpg')
d = data[data['male']].hist(column='bone_age_zscore')
plt.title('Male Distribution of Z-Score')
plt.savefig('./plots/male_bone_age.jpg')
e = data[data['female']].hist(column='bone_age_zscore')
plt.title('Female Distribution of Z-Score')
plt.savefig('./plots/female_bone_age.jpg')

