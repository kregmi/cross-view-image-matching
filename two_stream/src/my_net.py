from VGG import VGG16
from siamese_fc import Siamese_FC

import tensorflow as tf


def two_stream_baseline(x_sat, x_grd, keep_prob, trainable):
    with tf.device('/gpu:1'):
        vgg_grd = VGG16()
        grd_local = vgg_grd.eight_layer_conv_multiscale(x_grd, keep_prob, trainable, 'VGG_grd')

        vgg_sat = VGG16()
        sat_local = vgg_sat.eight_layer_conv_multiscale(x_sat, keep_prob, trainable, 'VGG_sat')

    with tf.device('/gpu:0'):
        fc = Siamese_FC()
        sat_global, grd_global = fc.my_siamese_fc_multiscale(sat_local, grd_local, trainable, 'dim_reduction')
    return sat_global, grd_global

