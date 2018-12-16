from keras.models import load_model
import numpy as np
from PIL import Image
from network import *
from utils import *
from hyperparametersearch import search
import argparse
import os, sys
from tqdm import tqdm

def get_params():
    return search('',0)


'''
This test file allows you to extract the super resolved image in order to see what sort of information has been extracted from the process.

'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-td","--dir",required=True,
            help='Path to Directory')
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
    model = ResnetBuilder._build_full(inp,gender,network=args['network'])
    model = ResnetBuilder.compile(model,*compile_params)
    model.summary()
    
    sr = True 
    weight_path = './model_weights/reg_{}_bone_age_weights.best.hdf5'.format(args['network'])    
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        print("Model loaded")
    else:
        sys.exit(0)
    super_res = model.layers[11].output
    resized = model.layers[10].output
    new_model = Model(inputs=[inp,gender],outputs=[super_res,resized]) 
    test = get_test(args['dir'])
    pbar = tqdm(test.iterrows())
    for i, row in pbar:
        label = row['boneage']
        path = row['path']
        gender = [1.0] if row['female'] else [0.0]
        image = open_image(path,dim=params['dim'][:2])
        image = np.expand_dims(normalize(image),-1)
        gender = np.expand_dims(np.array(gender),0)
        out_val = new_model.predict([np.expand_dims(image,0),gender])
        out = np.interp(out_val[0][0,:,:,0],(out_val[0].min(),out_val[0].max()),(0.0,1.0))
        out = np.uint8(out*255) 
        out_hand = np.interp(out_val[1][0,:,:,0],(out_val[1].min(),out_val[1].max()),(0.0,1.0))
        out_hand = np.uint8(out_hand*255)
        img = Image.fromarray(out,'L')
        img.save('super_res.jpg')
        img_hand = Image.fromarray(out_hand,'L')
        img_hand.save('hand.jpg')
        break

