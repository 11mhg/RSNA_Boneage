import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda

def _phaseshift(x,r=2):
    bsize, a, b, c = x.get_shape().as_list()
    bsize = tf.shape(x)[0]
    x = tf.reshape(x,(bsize,a,b,r,r))
    x = tf.transpose(x,(0,1,2,4,3))
    x = tf.split(x,a,axis=1)
    x = tf.concat([tf.squeeze(X,axis=1) for X in x],2)
    x = tf.split(x,b,axis=1)
    x = tf.concat([tf.squeeze(X, axis=1) for X in x],2)
    return tf.reshape(x, (bsize, a*r, b*r, 1))

def PhaseShift(x,r=2,color=False):
    if color:
        Xc = tf.split(x,3,3)
        X = tf.concat([_phaseshift(x, r) for x in Xc],3)
    else:
        X = _phaseshift(x,r)
    return X
    
def SubConv(X,r=2,color=False):
    _, _, _, c = X.get_shape().as_list()
    X = tf.layers.conv2d(X, 32,[3,3],activation=tf.nn.relu)
    X = tf.layers.conv2d(X, 32,[3,3],activation=tf.nn.relu)
    X = PhaseShift(X,r=2,color=color)
    return X


def SubpixelConv2D(input_shape,scale=4):
    def subpixel_shape(input_shape):
        dims=[input_shape[0],
              input_shape[1]*scale,
              input_shape[2]*scale,
              int(input_shape[3]/(scale**2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return PhaseShift(x,r=scale)

    return Lambda(subpixel,output_shape=subpixel_shape,name='subpixel') 


if __name__ == "__main__":
    with tf.Session() as sess:
        x = np.arange(2*16*16).reshape(2, 8, 8, 4)
        X = tf.placeholder(shape=(2, 8, 8, 4), name="X")# tf.Variable(x, name="X")
        Y = PhaseShift(X, 2, color=False)
        y = sess.run(Y, feed_dict={X: x})
        plt.imshow(y[0,:,:,0],interpolation="none")
        plt.show()

        x2 = np.arange(2*3*16*16).reshape(2, 8, 8, 4*3)
        X2 = tf.placeholder(shape=(2, 8, 8, 4*3), name="X")# tf.Variable(x, name="X")
        Y2 = PhaseShift(X2, 2,color=True)
        y2 = sess.run(Y2, feed_dict={X2: x2})
    y2 = np.interp(y2,(y2.min(),y2.max()),(0,255))
    plt.imshow(y2[0, :, :, 0], interpolation="none")
    plt.show()
