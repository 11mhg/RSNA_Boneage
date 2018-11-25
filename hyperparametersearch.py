from network import *
from utils import *
import keras.backend as K



def network_search():
    train_generator = RSNAGenerator(batch_size=32,dim=(224,224),train=True,shuffle=True)
    val_generator=RSNAGenerator(batch_size=32,dim=(224,224),train=False,shuffle=True)

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
                    print("%.4f percent done"%(count/total*100))
                    count+=1
                    K.clear_session()
                    reg_model, sup_model = ResnetBuilder.build((224,224,1),
                        scale = scale,
                        reg_params = compile_params,
                        sup_params = compile_params,
                        rep=rep,
                        bottle=bottle,
                        reg_param=reg_param)
                    reg_hist = reg_model.fit_generator(train_generator,epochs=1,callbacks=None,
                        validation_data=val_generator)
                    if sum(reg_hist.history['val_loss'])/len(reg_hist.history['val_loss']) < best:
                        best = sum(reg_hist.history['val_loss'])/len(reg_hist.history['val_loss']) 
                        best_rep = rep
                        best_bottle=bottle
                        best_reg_param=reg_param
                        best_scale = scale
    print("The following achieved the best validation loss in 4 epochs: \n",
            "rep: ", best_rep,
            "\nbottle: ", best_bottle,
            "\nreg_param: ", best_reg_param,
            "\nscale: ",best_scale)
    return {'rep':best_rep,'bottle':best_bottle,'reg':best_reg_param,'scale':best_scale}

def optimizer_search(model_params):
    train_generator = RSNAGenerator(batch_size=32,dim=(224,224),train=True,shuffle=True)
    val_generator = RSNAGenerator(batch_size=32,dim=(224,224),train=False,shuffle=True)

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
                    print("%.4f percent done."%(count/total*100))
                    compile_params=(custom_mae_metric,lr,b1,b2,None,0.0,amsgrad)
                    reg_model, sup_model = ResnetBuilder.build((224,224,1),
                            scale=model_params['scale'],
                            reg_params=compile_params,
                            sup_params=compile_params,
                            rep=model_params['rep'],
                            bottle=model_params['bottle'],
                            reg_param=model_params['reg'])
                    reg_hist = reg_model.fit_generator(train_generator,epochs=4,callbacks=None,
                            validation_data=val_generator)
                    if reg_hist.history['val_loss'][-1] < best:
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


def train_search(params):
    train_generator = RSNAGenerator(batch_size=32,dim=(224,224),train=True,shuffle=True)
    val_generator = RSNAGenerator(batch_size=32,dim=(224,224),train=True,shuffle=True)

    batch_sizes = [2,4,8,16,32,64,128]
    dims = [(224,224,1),(260,260,1),(384,384,1)]
    compile_params = (custom_mae_metric,params['lr'],params['b1'],params['b2'],None,0.0,params['amsgrad'])
    
    best = np.inf
    best_bs = None
    best_dim = None
    count = 0
    total = len(batch_sizes)*len(dims)

    for bs in batch_sizes:
        for dim in dims:
            print("%.4f percent done."%(count/total*100))
            count+=1
            K.clear_session()
            train_generator.batch_size=bs
            val_generator.batch_size=bs
            train_generator.dim = dim[:2]
            val_generator.dim = dim[:2]
            reg_model, sup_model = ResnetBuilder.build(
                    dim,
                    scale=model_params['scale'],
                    reg_params=compile_params,
                    sup_params=compile_params,
                    rep = params['rep'],
                    bottle=params['bottle'],
                    reg_param=params['reg'])
            reg_hist = reg_model.fit_generator(train_generator,epochs=4,callbacks=None,
                    validation_data=val_generator)
            if reg_hist.history['val_loss'][-1] < best:
                best = reg_hist.history['val_loss'][-1]
                best_bs = bs
                best_dim = dim
    print("The following are the best values for the train: \n",
            "batch size: ",best_bs,
            "\ndims: ",best_dim)
    return params.update({"bs":best_bs,"dim":best_dim})


def search():
    params = network_search()
#    params = optimizer_search(params)
#    params = train_search(params)
    return params


if __name__=="__main__":
    params = search()
    print(params)