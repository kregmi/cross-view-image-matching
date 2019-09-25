import cv2
import random
import numpy as np
import os

class InputData:


    img_root = './Dataset/cvusa_orig/'

    def __init__(self):

        self.train_list = self.img_root + 'splits/cvusa_edgemap_30_train.txt'
        self.test_list = self.img_root + 'splits/cvusa_edgemap_30_val.txt'




        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                line = line.rstrip('\n')
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                self.id_list.append([data[0], data[1], data[2], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)




        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                line = line.rstrip('\n')
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                self.id_test_list.append([data[0], data[1], data[2], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)




    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        
        batch_sat = np.zeros([batch_size, 512, 512, 6], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype = np.float32)
       


        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
            #print self.id_test_list[img_idx][0]
            # satellite
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
#            batch_sat[i, :, :, :] = img


            synth = cv2.imread(self.img_root + self.id_test_list[img_idx][2])
            synth = synth.astype(np.float32)
            synth[:, :, 0] -= 103.939  # Blue
            synth[:, :, 1] -= 116.779  # Green
            synth[:, :, 2] -= 123.6  # Red

            img_synth = np.concatenate((img, synth), 2)
            batch_sat[i, :, :, :] = img_synth




            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

        self.__cur_test_id += batch_size

        return batch_sat, batch_grd



    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None
        
        batch_sat = np.zeros([batch_size, 512, 512, 6], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype = np.float32)
     
        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # satellite
            img = cv2.imread(self.img_root + self.id_list[img_idx][0])
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i), img.shape)
                continue
            rand_crop = random.randint(1, 748)
            if rand_crop > 512:
                start = int((750 - rand_crop) / 2)
                img = img[start : start + rand_crop, start : start + rand_crop, :]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            rand_rotate = random.randint(0, 4) * 90
            rot_matrix = cv2.getRotationMatrix2D((256, 256), rand_rotate, 1)
            img = cv2.warpAffine(img, rot_matrix, (512, 512))
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6    # Red


            # satellite synthesized img
            synth = cv2.imread(self.img_root + self.id_list[img_idx][2])
            if synth is None or synth.shape[0] != synth.shape[1]:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][2], i), synth.shape)
                exit()

            rot_matrix = cv2.getRotationMatrix2D((256, 256), rand_rotate, 1)
            synth = cv2.warpAffine(synth, rot_matrix, (512, 512))
            synth = synth.astype(np.float32)
            # img -= 100.0
            synth[:, :, 0] -= 103.939  # Blue
            synth[:, :, 1] -= 116.779  # Green
            synth[:, :, 2] -= 123.6    # Red

            img_synth = np.concatenate((img, synth), 2)
            batch_sat[batch_idx, :, :, :] = img_synth

            # ground           
            img = cv2.imread(self.img_root + self.id_list[img_idx][1])
            if img is None or img.shape[0] != 224 or img.shape[1] != 1232:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][1], i), img.shape)
                continue
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red

            batch_grd[batch_idx, :, :, :] = img
            batch_idx += 1

        self.__cur_id += i
        return batch_sat, batch_grd


    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0

