from network import *
from utils import *
import keras.backend as K
import time
import argparse
import os,sys


def network_search(directory):
    train_generator = RSNAGenerator(directory,batch_size=32,dim=(224,224),train=True,shuffle=True)
    val_generator=RSNAGenerator(directory,batch_size=32,dim=(224,224),train=False,shuffle=True)

    possible_rep = [[i,j,k] for i in range(1,3) for j in range(2,8,2) for k in range(1,3)]
    possible_bottle = [True,False]
    possible_reg_param = [1.*pow(10,(-i)) for i in range(2,6,1)]
    scales = [2]


    best = np.inf
    best_rep = None
    best_bottle = None
    best_reg_param = None
    best_scale = None
    total = len(possible_rep)*len(possible_bottle)*len(possible_reg_param)*len(scales)
    compile_params=(custom_mae_metric,0.001,0.9,0.99,None,0.0,False)
    count = 0
    for rep in possible_rep:
        for bottle in possible_bottle:
            for reg_param in possible_reg_param:
                for scale in scales:
                    try:
                        count+=1
                        K.clear_session()
                        reg_model, sup_model = ResnetBuilder.build((224,224,1),
                            scale = scale,
                            reg_params = compile_params,
                            sup_params = compile_params,
                            rep=rep,
                            bottle=bottle,
                            reg_alph=reg_param)
                        reg_hist = reg_model.fit_generator(train_generator,epochs=1,callbacks=None,
                            validation_data=val_generator)
                        if sum(reg_hist.history['val_loss'])/len(reg_hist.history['val_loss']) < best:
                            best = sum(reg_hist.history['val_loss'])/len(reg_hist.history['val_loss']) 
                            best_rep = rep
                            best_bottle=bottle
                            best_reg_param=reg_param
                            best_scale = scale
                    except:
                        print("Combination causes error, continuing")
                        continue
    print("The following achieved the best validation loss in 4 epochs: \n",
            "rep: ", best_rep,
            "\nbottle: ", best_bottle,
            "\nreg_param: ", best_reg_param,
            "\nscale: ",best_scale)
    return {'rep':best_rep,'bottle':best_bottle,'reg':best_reg_param,'scale':best_scale}

def optimizer_search(model_params,directory):
    train_generator = RSNAGenerator(directory,batch_size=32,dim=(224,224),train=True,shuffle=True)
    val_generator = RSNAGenerator(directory,batch_size=32,dim=(224,224),train=False,shuffle=True)

    possible_lr = [1.*pow(10,-1) for i in range(1,4)]
    possible_b1 = [0.875+(0.015*i) for i in range(0,5)]
    possible_b2 = [min(0.919+(0.015*i),0.999) for i in range(0,7)]
    possible_b2 = list(set(possible_b2))
    possible_amsgrad = [True,False]

    best = np.inf
    best_lr = None
    best_b1 = None
    best_b2 = None
    best_amsgrad = None
    total = len(possible_lr) *len(possible_b1)*len(possible_b2)*len(possible_amsgrad)
    count = 0
    for lr in possible_lr:
        for b1 in possible_b1:
            for b2 in possible_b2:
                for amsgrad in possible_amsgrad:
                    count+=1
                    K.clear_session()
                    compile_params=(custom_mae_metric,lr,b1,b2,None,0.0,amsgrad)
                    reg_model, sup_model = ResnetBuilder.build((224,224,1),
                            scale=model_params['scale'],
                            reg_params=compile_params,
                            sup_params=compile_params,
                            rep=model_params['rep'],
                            bottle=model_params['bottle'],
                            reg_alph=model_params['reg'])
                    reg_hist = reg_model.fit_generator(train_generator,epochs=1,callbacks=None,
                            validation_data=val_generator)
                    if sum(reg_hist.history['val_loss'])/len(reg_hist.history['val_loss']) < best:
                        best = reg_hist.history['val_loss'][-1]
                        best_lr = lr
                        best_b1 = b1
                        best_b2 = b2
                        best_amsgrad = amsgrad
    print("The following are the best values for the opimizer: \n",
            "lr: ",best_lr,
            "\nb1: ",best_b1,
            "\nb2: ",best_b2,
            "\namsgrad: ",best_amsgrad)
    return model_params.update({"lr":best_lr,"b1":best_b1,"b2":best_b2,"amsgrad":best_amsgrad})


def train_search(params,directory,ind):
    train_generator = RSNAGenerator(directory,batch_size=32,dim=(224,224),train=True,shuffle=True)
    val_generator = RSNAGenerator(directory,batch_size=32,dim=(224,224),train=False,shuffle=True)
    G = len(get_available_gpus())
    batch_sizes = [2,4,8,16,32,64,128]
    dims = [(224,224,1),(260,260,1),(384,384,1)]
    compile_params = (custom_mae_metric,params['lr'],params['b1'],params['b2'],None,0.0,params['amsgrad'])
    
    best = np.inf
    best_bs = None
    best_dim = None
    count = 0
    total = len(batch_sizes)*len(dims)
    combo = [[bs,dim] for bs in batch_sizes for dim in dims]
    dim = combo[ind-1][1]
    bs = combo[ind-1][0]

#    for bs in batch_sizes:
#        for dim in dims:
    try:
        print(dim,bs) 
        start = time.time()
        count+=1
        print(count)
        K.clear_session()
        train_generator.batch_size=bs * G
        val_generator.batch_size=bs * G
        train_generator.dim = dim[:2]
        val_generator.dim = dim[:2]
        reg_model, sup_model = ResnetBuilder.build(
            dim,
            scale=params['scale'],
            reg_params=compile_params,
            sup_params=compile_params,
            rep = params['rep'],
            bottle=params['bottle'],
            reg_alph=params['reg'])
        reg_hist = reg_model.fit_generator(train_generator,epochs=1,callbacks=None,
            validation_data=val_generator)
        if sum(reg_hist.history['val_loss'])/len(reg_hist.history['val_loss']) < best:
            best = reg_hist.history['val_loss'][-1]
            best_bs = bs
            best_dim = dim
    except:
            print(dim,bs)
            print(params)
            print("Caused errors.") 
    print("The following are the best values for the train: \n",
            "batch size: ",best_bs,
            "\ndims: ",best_dim)
    with open('{}_search.out'.format(ind),'w+') as f:
        f.write('id: {}\nloss: {}\nbs: {}\ndim: {}'.format(ind,best,best_bs,best_dim))
    return params.update({"bs":best_bs,"dim":best_dim})


def search(directory,ind):
#    params = network_search(directory)
    params = {'scale': 2, 'rep': [1, 3, 2], 'reg': 0.01, 'bottle': True} 
#    params = optimizer_search(params,directory)
    params.update({"lr":  0.0001,"b1":  0.89,"b2":  0.934,"amsgrad":  False})

#    params = train_search(params,directory,ind)
    params.update({"bs":16,"dim":(260,260,1)})
    print("The parameters found were: ",params)
    return params


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-td","--dir",required=True,
            help="Path to directory")
    args = vars(ap.parse_args())
    params = search(args['dir'],int(os.environ['SLURM_NODEID']))
    print(params)
