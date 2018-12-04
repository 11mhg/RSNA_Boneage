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
    args = vars(ap.parse_args())
    params = get_params() 
    scaled_dim = (params['dim'][0]*2,params['dim'][1]*2,params['dim'][2])
    train_generator = RSNAGenerator(args['dir'],batch_size=params['bs'],
                      dim=scaled_dim[:2],train=True,shuffle=True)
    val_generator = RSNAGenerator(args['dir'],batch_size=params['bs'],
                      dim = scaled_dim[:2],train=False,shuffle=True)  

    compile_params = (custom_mae_metric,params['lr'],params['b1'],params['b2'],None,0.0,params['amsgrad'])

#    inp = Input(scaled_dim)
#    reg_model = ResnetBuilder._build_resnet50(inp)

#    reg_model = ResnetBuilder.compile(reg_model,*compile_params)
#    reg_model.summary()
#    if os.path.exists('./model_weights/reg_bone_age_weights.best.hdf5'):
#        reg_model.load_weights('./model_weights/reg_bone_age_weights.best.hdf5')
    #reg_model = train_reg(reg_model,train_generator,val_generator,epochs=int(args['epochs']))

    train_generator.dim = params['dim'][:2]
    val_generator.dim = params['dim'][:2]
    train_generator.batch_size=4
    val_generator.batch_size=4
    print("Refreshing generator and dims")
    train_generator.prep()
    val_generator.prep()

    compile_params = (custom_mae_metric,0.001,params['b1'],params['b2'],None,0.0,params['amsgrad'])

    print("Beginning super_res training")
#    sup_model = ResnetBuilder.build_sup(params['dim'],reg_model,compile_params, params['scale'])
#    sup_model = train_sup(sup_model,train_generator,val_generator,epochs=int(args['epochs']))

#    del sup_model
    reg_model = load_model('./reg_model.h5',custom_objects={'custom_mae_metric':custom_mae_metric})
    sup_model = ResnetBuilder.build_sup(params['dim'],reg_model,compile_params, params['scale'],trainable=True)
#    if os.path.exists('./model_weights/sup_bone_age_weights.best.hdf5'):
#        sup_model.load_weights('./model_weights/sup_bone_age_weights.best.hdf5')
    sup_model = train_sup(sup_model,train_generator,val_generator,epochs=int(args['epochs']))
