import cv2
import random
import numpy as np
import scipy.io as sio


class InputData:

    def __init__(self):

#        self.feat_path = self.img_root + './auxgan_II_train.mat'
        self.feat_path = '../../joint_feat_learning/src/joint_feat_learning_train.mat'
        feats = sio.loadmat(self.feat_path)
        self.grd_global_descriptor = feats['grd_feats']
        self.sat_global_descriptor = feats['sat_feats']
        self.gan_sat_global_descriptor = feats['gan_sat_feats']


        self.data_size1 = self.grd_global_descriptor.shape[0]
        self.__cur_id = 0  # for training
        self.id_idx_list = np.arange(self.data_size1)

        print('Train Data path: ' + self.feat_path)
        print('Train Data Size =', self.data_size1)



        self.feat_path_test =  '../../joint_feat_learning/src/joint_feat_learning_val.mat'
        feats_test = sio.loadmat(self.feat_path_test)
        self.grd_global_descriptor_test = feats_test['grd_feats']
        self.sat_global_descriptor_test = feats_test['sat_feats']
        self.gan_sat_global_descriptor_test = feats_test['gan_sat_feats']


        self.data_size1_test = self.grd_global_descriptor_test.shape[0]
        self.__cur_test_id = 0  
        self.id_test_idx_list = np.arange(self.data_size1_test)

        print('Test Data path: ' + self.feat_path_test)
        print('Test Data Size: ', self.data_size1_test)



    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.data_size1_test:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.data_size1_test:
            batch_size = self.data_size1_test - self.__cur_test_id

        batch_grd = np.zeros([batch_size, 2000], dtype = np.float32)
        batch_sat = np.zeros([batch_size, 1000], dtype = np.float32)


        feat1_sat = np.zeros([1, 1000], dtype = np.float32)
        feat1_grd = np.zeros([1, 1000], dtype = np.float32)

#        feat2_sat = np.zeros([1, 1000], dtype = np.float32)
        feat2_grd = np.zeros([1, 1000], dtype = np.float32)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
            # satellite
            feat1_sat = self.sat_global_descriptor_test[img_idx]
#            feat2_sat = self.sat_global_descriptor_test[img_idx]
#            batch_sat[i] = np.concatenate((feat1_sat, feat2_sat), axis=0)
            batch_sat[i] = feat1_sat

            # ground
            feat1_grd = self.grd_global_descriptor_test[img_idx]
            feat2_grd = self.gan_sat_global_descriptor_test[img_idx]
            batch_grd[i] = np.concatenate((feat1_grd, feat2_grd), axis=0)


        self.__cur_test_id += batch_size

        return batch_sat, batch_grd




    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size1:
            self.__cur_id = 0
            return None, None


        batch_grd = np.zeros([batch_size, 2000], dtype = np.float32)
        batch_sat = np.zeros([batch_size, 1000], dtype = np.float32)
#        batch_sat = np.zeros([batch_size, 2000], dtype = np.float32)

        feat1_sat = np.zeros([1, 1000], dtype = np.float32)
        feat1_grd = np.zeros([1, 1000], dtype = np.float32)

#        feat2_sat = np.zeros([1, 1000], dtype = np.float32)
        feat2_grd = np.zeros([1, 1000], dtype = np.float32)

        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size1:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # satellite
            feat1_sat = self.sat_global_descriptor[img_idx]
#            feat2_sat = self.sat_global_descriptor[img_idx]
            batch_sat[batch_idx] = feat1_sat
#            batch_sat[batch_idx] = np.concatenate((feat1_sat, feat2_sat), axis=0)

            # ground
            feat1_grd = self.grd_global_descriptor[img_idx]
            feat2_grd = self.gan_sat_global_descriptor[img_idx]
            batch_grd[batch_idx] = np.concatenate((feat1_grd, feat2_grd), axis=0)


            batch_idx += 1

        self.__cur_id += i

        return batch_sat, batch_grd





    def get_dataset_size(self):
        return self.data_size1

    def get_test_dataset_size(self):
        return self.data_size1_test

    def reset_scan(self):
        self.__cur_test_idd = 0

