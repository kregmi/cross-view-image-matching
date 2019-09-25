from my_net import joint_feat_learning
from input_data import InputData

import tensorflow as tf
import numpy as np
import os

import scipy.io as sio


from tensorflow.python import pywrap_tensorflow

# --------------  configuration parameters  -------------- #
# the type of network to be used: "joint_feat_learning"
network_type = 'joint_feat_learning'

batch_size = 24
is_training = True
loss_weight = 10.0
number_of_epoch = 2

learning_rate_val = 1e-5
keep_prob_val = 0.5



# -------------------------------------------------------- #



def validate(grd_descriptor, sat_descriptor):
    
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))

# top 1% accuracy 
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

    # get variable list of pretrained model: source of pretrained weights
#    CHECKPOINT_NAME = '../../baseline_cvusa/Model/my_net_ms/model.ckpt'
#    restored_vars = get_tensors_in_checkpoint_file(file_name=CHECKPOINT_NAME)

    # import data
    input_data = InputData()


    # define placeholders
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
    sat_x_synth = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x_synth')
    
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)



    # build model: three_stream_with_gan_imgs
    if network_type == 'joint_feat_learning':
        sat_global, grd_global, gan_sat_global = joint_feat_learning(sat_x, grd_x, sat_x_synth, keep_prob, is_training)
    else:
        print ('CONFIG ERROR: wrong network type, only joint_feat_learning valid')


    # define loss
    loss1 = compute_loss(sat_global, grd_global, 0)
    loss2 = compute_loss(gan_sat_global, sat_global, 0)

#    loss = (loss1 + 10 * loss2)/(1 + 10)
    loss = ( 10*loss1 + loss2)/(11)


    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver_full = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    
    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()


##        # loading pretrained model and copying its weights: restoring the pretrained wts from two stream baseline.
#        var_list = tf.global_variables()
#        var_list = [x for x in var_list if str(x.name)[:-2] in restored_vars[0]]
#        saver = tf.train.Saver(var_list, max_to_keep=None)
#        print('load pretrained model...')
#        saver.restore(sess, CHECKPOINT_NAME)
#        print(" Pretrained model loaded from: %s" % CHECKPOINT_NAME)
#        print('load model...FINISHED')



        # # # load model from intermediate epoch of joint feature learning experiment
#        load_model_path = '../Model/' + network_type + '/model.ckpt'
#        saver_full.restore(sess, load_model_path)
#        print("   Model loaded from: %s" % load_model_path)
#        print('load model...FINISHED')



        # Train
        best_accuracy = 0.0
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat, batch_grd = input_data.next_pair_batch(batch_size)

                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat[:, :, :, :3], grd_x: batch_grd, sat_x_synth: batch_sat[:, :, :, 3:],
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val}
                if iter % 50 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: loss : %.4f' %
                          (global_step_val, epoch, iter, loss_val))
                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1
            
            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()
            sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 1000])
            gan_sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 1000])
            grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 1000])
            val_i = 0
            while True:
                if (val_i % 2000 == 0):      
                    print('      progress %d' % val_i)

                batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {sat_x: batch_sat[:, :, :, :3], grd_x: batch_grd, sat_x_synth: batch_sat[:, :, :, 3:], keep_prob: 1.0}
                sat_global_val, grd_global_val, gan_sat_global_val = \
                    sess.run([sat_global, grd_global, gan_sat_global], feed_dict=feed_dict)


                gan_sat_global_descriptor[val_i: val_i + gan_sat_global_val.shape[0], :] = gan_sat_global_val
                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]

            print('   compute gan+aerial accuracy')
            val_accuracy1 = validate(gan_sat_global_descriptor, sat_global_descriptor)
            print('   %d: accuracy = %.2f%%' % (epoch, val_accuracy1*100.0))


            print('   compute real+aerial accuracy')
            val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
            print('   %d: accuracy = %.2f%%' % (epoch, val_accuracy*100.0))
#            exit()
            with open('../Result/' + str(network_type) + '_accuracy.txt', 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')
            print('   %d: accuracy = %.2f%%' % (epoch, val_accuracy*100.0))



            model_dir = '../Model/' + network_type + '/'

            if (best_accuracy < val_accuracy):
                best_accuracy = val_accuracy
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                save_path = saver_full.save(sess, model_dir + 'model.ckpt')
                print("Model saved in file: %s" % save_path)
                sio.savemat(str(network_type) + '.mat', dict([('grd_feats', grd_global_descriptor), ('sat_feats', sat_global_descriptor), ('gan_sat_feats', gan_sat_global_descriptor)]))

            else:
                print("Model not saved for epoch:" + str(epoch))
            # ---------------------------------------------------------



if __name__ == '__main__':
    train()
