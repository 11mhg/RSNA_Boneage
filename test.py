from keras.models import load_model
import numpy as np
from network import *
from utils import *
from hyperparametersearch import search
import argparse
import os, sys
from tqdm import tqdm

def get_params():
    return search('',0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-td","--dir",required=True,
            help='Path to Directory')
    ap.add_argument('-sr','--super',required=True,
            help="Superresolution layer")
    ap.add_argument('-n','--network',required=True,
            help="vgg or resnet")
    args = vars(ap.parse_args())
    params = get_params() 
    scaled_dim = (params['dim'][0]*2,params['dim'][1]*2,params['dim'][2])
    
    train_generator = RSNAGenerator(args['dir'],batch_size=params['bs'],
            dim = params['dim'][:2],train=False,shuffle=False)
    bone_age_div = train_generator.bad
    bone_age_mean = train_generator.mean
    compile_params = (custom_mae_metric,params['lr'],params['b1'],params['b2'],None,0.0,params['amsgrad'])

    inp = Input(params['dim'])
    gender = Input((1,))
    if args['super'] == 'True':
        model = ResnetBuilder._build_full(inp,gender,network=args['network'])
    else:
        model = ResnetBuilder._build_full_nosr(inp,gender,network=args['network'])
    model = ResnetBuilder.compile(model,*compile_params)
    model.summary()
    
    sr = args['super']=='True'
    if sr:
        weight_path = './model_weights/reg_{}_bone_age_weights.best.hdf5'.format(args['network'])
    else:
        weight_path = './model_weights/reg_nosr_{}_weights.best.hdf5'.format(args['network'])
    print(weight_path)
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        print("Model loaded")
    else:
        sys.exit(0)

    test = get_test(args['dir'])
    mad = []
    rmse_y = []
    rmse_m = []
    pbar = tqdm(test.iterrows())
    for i, row in pbar:
        label = row['boneage']
        path = row['path']
        gender = [1.0] if row['female'] else [0.0]
        image = open_image(path,dim=params['dim'][:2])
        image = np.expand_dims(normalize(image),-1)
        gender = np.expand_dims(np.array(gender),0)
        out_val = model.predict([np.expand_dims(image,0),gender])
        rmse_y.append(((((out_val[0] * bone_age_div)+bone_age_mean)/12.)-(label/12.))**2)
        rmse_m.append(((((out_val[0] * bone_age_div)+bone_age_mean)/1.)-(label/1.))**2)
        mad.append(abs(((out_val[0]*bone_age_div) +bone_age_mean) - label))
    rmse_y = np.array(rmse_y)
    rmse_m = np.array(rmse_m)
    print("The testing MAD is: ",(sum(mad))/len(mad))
    mad = np.array(mad)
    print("The mad std is : ",mad.std())
    print("The rmse in years is : ",(np.sqrt(rmse_y.mean())))
    print("The rmse in months is : ",(np.sqrt(rmse_m.mean())))
