from siamese_fc import Siamese_FC
import tensorflow as tf



def one_layer_fc(x_sat, x_grd, keep_prob, trainable):
    with tf.device('/gpu:0'):
        fc = Siamese_FC()
        sat_global, grd_global = fc.one_layer_siamese_fc(x_sat, x_grd, trainable, 'fc')

    return sat_global, grd_global

