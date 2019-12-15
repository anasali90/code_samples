import cv2
from imgaug import augmenters as iaa
import random
import numpy as np
from keras.models import Sequential

#TODO Bootstrapping

class BatchGenerator (Sequential):

    def __init__(self,
                 id_data,
                 id_list,
                 val_id_list,
                 train_or_val,
                 batch_size,
                 shuffle,
                 epoch_size,

                 aug_factor): # let batch size equals to 10
        self.id_data = id_data
        self.indexes = id_list
        self.val_ID_list = val_id_list
        self.train_or_val = train_or_val
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch_size = epoch_size
        self.aug_factor = aug_factor
        if self.train_or_val == 'train':
            'Augmentation all data indexes at the begining of each' \
            ' epoch and before choosing the data for each batch'
            self.augment_indexes()

        self.match_indexes_pairs()
        self.on_epoch_end()
        self.index = 0

    def __len__(self):
        return int(np.floor(self.epoch_size / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self.indexes[0][self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return self[self.index]

    def __getitem__(self, idx):
        'Generate one batch of data with corresponding positive and negative patches percentage'
        # Generate indexes of the batch
        indexes_p1 = self.indexes[0][idx * self.batch_size:(idx + 1) * self.batch_size]
        indexes_p2 = self.indexes[1][idx * self.batch_size:(idx + 1) * self.batch_size]
        input1 = np.zeros([self.batch_size, 64, 32, 3])
        input2 = np.zeros([self.batch_size, 64, 32, 3])
        count = 0
        labels = np.zeros(shape=(self.batch_size, 2))
        for j, p1 in enumerate(indexes_p1):
            try:
                p2 = indexes_p2[j]
                idx_negative_pairs = j + int(self.batch_size / (2 + idx))
                img1 = self.id_data[p1[0], 0][p1[1], p1[2]]
                img2 = self.id_data[p2[0], 0][p2[1], p2[2]]
                if self.train_or_val == 'train':
                    img1 = self.preprocessing(img1)
                    img2 = self.preprocessing(img2)
                    img1 = self.apply_translation(img1, p1)
                    img2 = self.apply_translation(img2, p2)
                input1[count:count + 1, :, :, :] = img1
                input2[count:count + 1, :, :, :] = img2
                labels[count,0] = 1
                labels[count,1] = 0
                for i in range(idx + 1):
                    count += 1
                    p1 = indexes_p1[idx_negative_pairs]
                    p2 = indexes_p2[random.randint(0, 511)]
                    while p2[:2] == p1[:2]:
                        p2 = indexes_p2[random.randint(0, 511)]

                    img1 = self.id_data[p1[0], 0][p1[1], p1[2]]
                    img2 = self.id_data[p2[0], 0][p2[1], p2[2]]
                    if self.train_or_val == 'train':
                        img1 = self.preprocessing(img1)
                        img2 = self.preprocessing(img2)
                        img1 = self.apply_translation(img1, p1)
                        img2 = self.apply_translation(img2, p2)
                    input1[count:count + 1, :, :, :] = img1
                    input2[count:count + 1, :, :, :] = img2
                count += 1
            except IndexError:
                break
        c = list(zip(input1, input2, labels))
        random.shuffle(c)
        X1, X2, Y = zip(*c)
        return [np.asarray(X1), np.asarray(X2)], np.asarray(Y)

    def on_epoch_end(self):
        c = list(zip(self.indexes[0], self.indexes[1]))
        random.shuffle(c)
        self.indexes[0], self.indexes[1] = zip(*c)

    @staticmethod
    def preprocessing(img):
        # TODO make sure of this function
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        img_resized = cv2.resize(img, (32, 64))
        return img_resized

    def augment_indexes(self):
        for i in self.indexes:
            for j in i:
                a = len(j)
                for k in range(a) :
                    for l in range(self.aug_factor):
                        b = j[4*k] + [random.uniform(0, 0.05)] # adding aug translation percentage to indexes
                        j.insert((k*4+1)+l, b)

    @staticmethod
    def apply_translation(img, idx):
        if len(idx) ==4:
            seq = iaa.Sequential([
                iaa.Crop(percent=(idx[3], idx[3], idx[3], idx[3]))
            ])
            img = seq.augment_images(img)
        else:
            pass
        return img

    def match_indexes_pairs(self):
        lp1_ = [[self.indexes[0][i][j]] * len(self.indexes[1][i]) for i in range(len(self.indexes[0])) for j in
                range(len(self.indexes[0][i]))]
        lp1 = [val for sublist in lp1_ for val in sublist]
        lp2_ = [self.indexes[1][i] * len(self.indexes[0][i]) for i in range(len(self.indexes[0]))]
        lp2 = [val for sublist in lp2_ for val in sublist]
        self.indexes = [lp1, lp2]
