from network import *
from train_50 import *
from utils import *
from keras.models import *

def save_image(arr,name):
    from PIL import Image
    if arr.max() < 1:
        arr = np.uint8(arr*255)
    im = Image.fromarray(arr[:,:,0],'L')
    im.save(name)

if __name__ == '__main__':
    params = get_params()

    val_generator = RSNAGenerator('./',batch_size=1,
            dim = params['dim'][:2],train=False,shuffle=False)

#    model = load_model('./reg_model.h5',custom_objects={'custom_mae_metric':custom_mae_metric})
#    compile_params = (custom_mae_metric,params['lr'],params['b1'],params['b2'],None, 0.0, params['amsgrad'])
#    model = ResnetBuilder.build_sup(params['dim'],model,compile_params,params['scale'],trainable=False)
#    model.load_weights('./model_weights/sup_bone_age_weights.best.hdf5')
#    sup_out = model.get_layer('subpixel').output
#
#    model = Model(inputs = model.input,
#                  outputs= sup_out)
#    out_sample = model.predict(val_generator.__getitem__(0)[0])
#
#    if len(out_sample.shape) > 3:
#        out_sample = out_sample[0,:,:,:]
#    
#    save_image(out_sample,'subpixel.jpg')

    model = load_model('./reg_model.h5',custom_objects={'custom_mae_metric':custom_mae_metric})
    val_generator.dim=(520,520)
    val_generator.prep()
    inp = val_generator.__getitem__(0)[0]
    inp = inp[0,:,:,:]
    model.summary()
    attn_out = model.get_layer('multiply_1').output

    model = Model(inputs = model.input,
                  outputs = attn_out)
    out_sample = model.predict(val_generator.__getitem__(0)[0])
    if len(out_sample.shape)>3:
        out_sample = out_sample[0,:,:,:]
    save_image(out_sample,'attention.jpg')
    save_image(inp,'input_image.jpg')
