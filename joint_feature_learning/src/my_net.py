from VGG import VGG16
from siamese_fc import Siamese_FC

import tensorflow as tf


# ----------------------------------------------------------------------------------------------------

def joint_feat_learning(x_sat, x_grd, x_grd_gan, keep_prob, trainable):
    with tf.device('/gpu:1'):
        vgg_grd = VGG16()
        grd_local = vgg_grd.eight_layer_conv_multiscale(x_grd, keep_prob, trainable, 'VGG_grd')
        
        vgg_sat = VGG16()
        sat_local = vgg_sat.eight_layer_conv_multiscale(x_sat, keep_prob, trainable, 'VGG_sat')

        vgg_grd_gan = VGG16()
        grd_local_gan = vgg_grd_gan.eight_layer_conv_multiscale(x_grd_gan, keep_prob, trainable, 'VGG_grd_seg')
        
        
    with tf.device('/gpu:0'):
        fc = Siamese_FC()
        
        sat_global, grd_global, grd_global_gan = fc.three_stream_joint_feat_learning(sat_local, grd_local, grd_local_gan, trainable, 'dim_reduction')
    return sat_global, grd_global, grd_global_gan



