import model
import preprocessing

#TODO try to make a configurqation file to generlize your work
### for configuration file  ####

data_mat_dir = 'D:\RE_ID\cuhk03_release\\datatrain.npy'
id_train_or_val = 'D:\RE_ID\cuhk03_release\\train_or_val.npy'
train_or_val = 'train'
batch_size = 512
shuffle = True
epoch_size = 300*512
ids_train_num = 1164
####
a = model.FPNN()
a.train(data_mat_dir, id_train_or_val, train_or_val, batch_size, shuffle, epoch_size)

#preprocessing.parse_annotation(data_mat_dir,imgs_dir)

