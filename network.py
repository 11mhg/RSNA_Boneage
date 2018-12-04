import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Dropout, Concatenate, concatenate, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D 
from keras.layers.merge import add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
from keras.metrics import mean_absolute_error
from sklearn import metrics
from subpixel import SubpixelConv2D
from utils import custom_mae_metric
from tensorflow.python.client import device_lib
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import *
"""
This code is inspired from the keras resnet implementation
The keras implementation of resnet is from:
https://github.com/raghakot/keras-resnet
It has been modified to allow for easy parameter searching and compilation/training
"""

regularizer_param = 1.e-4

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type =='GPU']



def _bn_relu(inp):
    norm = BatchNormalization(axis=-1)(inp)
    return Activation("relu")(norm)

def _conv_bn_relu(filters, kernel_size, strides=(1,1), kernel_initializer="he_normal",padding="same"):
    global regularizer_param
    def func(inp):
        conv = Conv2D(filters=filters,kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2(regularizer_param))(inp)
        return _bn_relu(conv)
    return func

def _bn_relu_conv(filters, kernel_size, strides=(1,1), kernel_initializer="he_normal",padding="same"):
    global regularizer_param
    def func(inp):
        activation = _bn_relu(inp)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2(regularizer_param))(activation)
    return func

def _shortcut(inp,residual):
    #adds shortcut between inp and residual and merges
    global regularizer_param
    inp_shape = K.int_shape(inp)[1:]
    residual_shape = K.int_shape(residual)[1:]

    
    stride_height = int(round(inp_shape[0]/residual_shape[0]))
    stride_width = int(round(inp_shape[1]/residual_shape[1]))
    eq_channels = inp_shape[2] == residual_shape[2]

    shortcut = inp

    if stride_width > 1 or stride_height > 1 or not eq_channels:
        shortcut = Conv2D(filters=residual_shape[2],
                          kernel_size=(1,1),
                          strides=(stride_width,stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(regularizer_param))(inp)
    return add([shortcut,residual])

def _res_block(block_function, filters, repetitions, is_first_layer=False):
    def f(inp):
        for i in range(repetitions):
            init_strides=(1,1)
            if i==0 and not is_first_layer:
                init_strides = (2,2)
            temp = is_first_layer and i==0
            inp = block_function(filters=filters, init_strides=init_strides, is_first_layer=temp)(inp)
        return inp
    return f

def basic_block(filters, init_strides=(1,1), is_first_layer=False):
    #Basic 3x3 conv block from resnet
    global regularizer_param
    def f(inp):
        if is_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3,3),
                    strides=init_strides,
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(regularizer_param))(inp)
        else:
            conv1 = _bn_relu_conv(filters=filters,kernel_size=(3,3),strides=init_strides)(inp)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3,3))(conv1)
        return _shortcut(inp,residual)
    return f

def bottleneck(filters, init_strides=(1,1), is_first_layer=False):
    global regularizer_param
    def f(inp):
        if is_first_layer:
            conv_1_1 = Conv2D(filters=filters,kernel_size=(1,1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(regularizer_param))(inp)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters,kernel_size=(1,1),
                                     strides=init_strides)(inp)
        
        conv_3_3 = _bn_relu_conv(filters=filters,kernel_size=(3,3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters*4, kernel_size=(1,1))(conv_3_3)
        return _shortcut(inp,residual)
    return f





class ResnetBuilder(object):
    @staticmethod
    def _build_regressor(inp, m1_out, repetitions, bottle=False, filt_start = 64, kernel_1 = (7,7), stride_start = (2,2),reg_param=1.e-4):
        '''
        Builds a custom resnet like architecture
        '''
        if not bottle:
            block_fn = basic_block
        else:
            block_fn = bottleneck
        global regularizer_param
        regularizer_param = reg_param
        conv1 = _conv_bn_relu(filters = filt_start, kernel_size=kernel_1, strides=stride_start)(m1_out)
        pool1 = MaxPooling2D(pool_size=tuple([i//2 for i in kernel_1]),strides=(2,2),padding="same")(conv1)

        block = pool1
        filters = filt_start 
        for i, r in enumerate(repetitions):
            block = _res_block(block_fn,filters=filters, repetitions=r, is_first_layer=i==0)(block)
            filters*=2
        block = _bn_relu(block)

        #classifier_block
        block_shape = K.int_shape(block)[1:]
        pool2 = AveragePooling2D(pool_size=(block_shape[0],block_shape[1]),strides=(1,1))(block)
            
        flatten1= Flatten()(pool2)
        dense1 = Dense(units=1024, kernel_initializer="he_normal",
            activation='tanh')(flatten1)
        dense1 = Dropout(0.25)(dense1)
        dense2 = Dense(units=1, kernel_initializer="he_normal",
            activation='linear')(dense1)

        model = Model(inputs=inp, outputs=dense2)
        return model
    
    def _build_resnet50(inp):
        inp_conc = concatenate([inp,inp,inp])
        m = keras.applications.ResNet50(include_top = False,weights='imagenet',input_shape=inp_conc.get_shape().as_list()[1:],pooling=None)
        m.trainable = False
        m_out = m(inp_conc)
        attn_layer1 = Conv2D(2048,kernel_size=[1,1],activation='sigmoid')(m_out)
        attn_layer2 = Conv2D(2048,kernel_size=[1,1],activation='linear',use_bias=False)(attn_layer1)
        attn_layer3 = Multiply()([m_out,attn_layer2])
        attn_layer3 = Lambda(lambda x: x,name='attention')(attn_layer3)
        flat = Flatten()(attn_layer3)
        fcl = Dense(256,activation='linear')(flat)
        out = Dense(1,activation='linear')(fcl)
        return Model(inputs=inp,outputs=out)


    @staticmethod
    def build_sup(input_shape,reg_model,sup_params,scale,trainable=False):
        max_filt = scale ** 2
        gpus = get_available_gpus()
        G = len(gpus)

        if G <=1:
            print("One gpu found, training as normal")
            inp = Input(shape=input_shape)
            conv1 = _conv_bn_relu(filters=32,kernel_size=(3,3),strides=(1,1))(inp)
            conv2 = _conv_bn_relu(filters=64,kernel_size=(3,3),strides=(1,1))(conv1)
            conv3 = _conv_bn_relu(filters=max_filt,kernel_size=(3,3),strides=(1,1))(conv2)
            sub = SubpixelConv2D(input_shape=input_shape,scale=scale)(conv3)
            for layer in reg_model.layers:
                layer.trainable=trainable
            reg_model.layers.pop(0)
            new_out = reg_model(sub)
            sup_model = Model(inputs=inp,outputs=new_out)
        else:
            print("Training with {} GPUS...".format(G))
            
            with tf.device("/cpu:0"):
                inp = Input(shape=input_shape)
                conv1 = _conv_bn_relu(filters=32, kernel_size=(3,3), strides=(1,1))(inp)
                conv2 = _conv_bn_relu(filters=64, kernel_size=(3,3), strides=(1,1))(conv1)
                conv3 = _conv_bn_relu(filters=max_filt,kernel_size=(3,3),strides=(1,1))(conv2)
                sub = SubpixelConv2D(input_shape=input_shape,scale=scale)(conv3)
                for layer in reg_model.layers:
                    layer.trainable=trainable
                reg_model.layers.pop(0)
                new_out = reg_model(sub)
                sup_model=Model(inputs=inp,outputs=new_out)
            sup_model = multi_gpu_model(sup_model,gpus=G)
        sup_model = ResnetBuilder.compile(sup_model,*sup_params)
        sup_model.summary()
        return sup_model

    @staticmethod
    def compile(model,metric,lr=0.001,b1=0.9,b2=0.999,min_delta=None,decay=0.0,amsgrad=False):
        optimizer = keras.optimizers.Adam(lr,b1,b2,min_delta,decay,amsgrad)

        model.compile(optimizer=optimizer,loss='mse', metrics = [metric])
        
        return model
    

def train_sup(m1, train_generator, val_generator,epochs=10):
    weight_path = "./model_weights/sup_bone_age_weights.best.hdf5"
    checkpoint = ModelCheckpoint(weight_path,monitor='val_loss',verbose=1,
            save_best_only=True,mode='min',save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=10,verbose=1,mode='auto',min_delta=0.0001, cooldown=5,min_lr=0.00001)
    early = EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=10)
    tb = TensorBoard(log_dir='./logs/sup',histogram_freq=epochs,batch_size=train_generator.batch_size,
            write_graph=True,write_images=False)
    callbacks = [checkpoint,reduceLROnPlat,early]

    H = m1.fit_generator(train_generator,epochs=epochs,validation_data=val_generator,callbacks=callbacks)

    history = H.history
    for key in history.keys():
        plt.plot(history[key])
        plt.title(str(key))
        plt.xlabel('epoch')
        plt.savefig('./plots/'+str(key)+'-plot.png')
        plt.close()

    m1.save('sup_model.h5')
    print("SubPixel network trained")
    return m1

def train_reg(m1, train_generator, val_generator,epochs=10):
    weight_path = "./model_weights/reg_bone_age_weights.best.hdf5"
    checkpoint = ModelCheckpoint(weight_path,monitor='val_loss',verbose=1,
            save_best_only=True,mode='min',save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=10,verbose=1,mode='auto',min_delta=0.0001, cooldown=5,min_lr=0.00001)
    early = EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=10)
#    tb = TensorBoard(log_dir='./logs',histogram_freq=epochs,batch_size=train_generator.batch_size,
#            write_graph=True,write_images=False)

    callbacks = [checkpoint,reduceLROnPlat,early]

    H = m1.fit_generator(train_generator,epochs=epochs,validation_data=val_generator,callbacks=callbacks)

    history = H.history
    for key in history.keys():
        plt.plot(history[key])
        plt.title(str(key))
        plt.xlabel('epoch')
        plt.savefig('./plots/'+str(key)+'-plot.png')
        plt.close()
    m1.save('reg_model.h5')
    print("Regularization network trained")
    return m1
