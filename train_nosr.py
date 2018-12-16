from keras.models import load_model
from network import *
from utils import *
from hyperparametersearch import search
import argparse
import os, sys

def get_params():
    return search('',0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-td","--dir",required=True,
            help='Path to Directory')
    ap.add_argument("-e","--epochs",required=True,
            help="Number of total epochs")
    ap.add_argument('-n','--network',required=True,
            help='vgg or resnet')
    args = vars(ap.parse_args())
    params = get_params() 
    scaled_dim = (params['dim'][0]*2,params['dim'][1]*2,params['dim'][2])
    train_generator = RSNAGenerator(args['dir'],batch_size=params['bs'],
                      dim=params['dim'][:2],train=True,shuffle=True)
    val_generator = RSNAGenerator(args['dir'],batch_size=params['bs'],
                      dim = params['dim'][:2],train=False,shuffle=True)  

    compile_params = (custom_mae_metric,params['lr'],params['b1'],params['b2'],None,0.0,params['amsgrad'])
    
    div = train_generator.bad
    mean = train_generator.mean

    inp = Input(params['dim'])
    gender = Input((1,))
    model = ResnetBuilder._build_full_nosr(inp,gender,network=args['network'])
    model = ResnetBuilder.compile(model,*compile_params)
    model.summary()
    model = train_reg(model,train_generator,val_generator,epochs=int(args['epochs']),sr=False,network=args['network'])
    test_df = get_test(args['dir'])
    mad = []

    for i, row in test_df.iterrows():
        label = row['boneage']
        path = row['path']
        gender = [1.0] if row['female'] else [0.0]
        img = open_image(path,params['dim'][:2])
        img = normalize(img)
        img = np.expand_dims(img,-1)
        gender = np.expand_dims(np.array(gender),0)
        val = model.predict([np.expand_dims(img,0),gender])
        mad.append(abs(label - (val[0]*div + mean)))
    print("Test MAD: ",(sum(mad)/len(mad)))

 
