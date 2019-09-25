from cvm_net import my_net_multiscale
from input_data import InputData

import tensorflow as tf
import numpy as np
import os
import scipy.io as sio



# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = 'two-stream_baseline_backup'

batch_size = 30
is_training = True
loss_weight = 10.0
number_of_epoch = 1

learning_rate_val = 1e-5
keep_prob_val = 0.8
# -------------------------------------------------------- #

def compute_accuracy(grd_descriptor, sat_descriptor, start_index, top1_percent):
    accuracy = 0.0
    data_amount = 0.0

    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    for i in range(grd_descriptor.shape[0]):
        gt_dist = dist_array[i, i]

        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy



def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))

    top10 = 10
    sio.savemat( str(network_type) + '_test_index_matrix.mat', dict([('dist_array', dist_array)]))
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top10:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    print 'Accuracy for top 10: '
    print accuracy



    accuracy = 0.0
    data_amount = 0.0
    top1 = 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    print 'Accuracy for top 1: '
    print accuracy



    accuracy = 0.0
    data_amount = 0.0
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    print 'Accuracy for top 1 percent: '
    print accuracy


    return accuracy


def compute_loss(sat_global, grd_global, batch_hard_count=0):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss


def train(start_epoch=1):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data
    input_data = InputData()


    # define placeholders
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)


    # build model
    if network_type == 'two-stream_baseline_backup':
        sat_global, grd_global = two_stream_baseline(sat_x, grd_x, keep_prob, is_training)
    else:
        print ('CONFIG ERROR: wrong network type, only two_stream_baseline valid')


    # define loss
    loss = compute_loss(sat_global, grd_global, 0)


    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')
        load_model_path = '../Model/' + network_type + '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')



        # ---------------------- validation ----------------------
        print('validate test set...')
        print('   compute global descriptors')
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 1000])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 1000])
        val_i = 0
        while True:
            if val_i % 100 == 0:
	    print('      progress %d' % val_i)

            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('   save the features   ')
        sio.savemat(str(network_type) + '.mat', dict([('grd_feats', grd_global_descriptor),('sat_feats', sat_global_descriptor)]))
        print('Features saved in: ' str(network_type) '.mat')

        print('   compute accuracy')
        val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
        print('   Test accuracy = %.2f%%' % (val_accuracy*100.0))

if __name__ == '__main__':
    train()
