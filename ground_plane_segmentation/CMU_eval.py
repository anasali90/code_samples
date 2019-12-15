
import numpy as np

from keras.models import load_model
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import cv2


'''
l = []
img2 = cv2.imread('cam1.png')
img2 = cv2.resize(img2,(320,240))
l.append(img2)
img3 = cv2.imread('cam2.png')
img3 = cv2.resize(img3,(320,240))
l.append(img3)
img4 = cv2.imread('cam3.png')
img4 = cv2.resize(img4,(320,240))
l.append(img4)
img5 = cv2.imread('cam4.png')
img5 = cv2.resize(img5,(320,240))
l.append(img5)
img6 = cv2.imread('cam5.png')
img6 = cv2.resize(img6,(320, 240))
l.append(img6)
img7 = cv2.imread('cam6.png')
img7 = cv2.resize(img7,(320,240))
l.append(img7)

a = np.asarray(l)
np.save('CMU_sample_images.npy',a)
'''
model11 = load_model('new_weights_6.hdf5')
model11.summary()
images = np.load('CAMNET_sample_images.npy')
for i in range(0, 7):
   img = images[i,:,:,:]
 #  img = cv2.resize(img, (260,340))
   im = img
   img = img.reshape([1, 240, 320, 3])


   nn = model11.predict_on_batch(img)
   predected_gp_label = (nn[:, :, :, 0] > 0.5) * 1

   labels_img = label(predected_gp_label)
   props = regionprops(labels_img)
   area = []
   for  p in props :
      area.append(p.area)
   coords = props[area.index(max(area))].coords
   box1 = coords[coords[:,0]<(coords[0,0]+20)]
   p1 = [ np.amin(box1[:,1]),box1[0,0] ]
   p2 = [np.amax(box1[:,1]), box1[0,0]]
   box2 = coords[coords[:,0]>(coords[-1,0]-20)]
   p3 = [ np.amin(box2[:,1]), box2[-1,0]]
   p4 = [np.amax(box2[:,1]), box2[-1,0]]
   '''
   projected_point = np.matmul(h, desored_point[x,y,1])
   projected_point_coords = projected_point/projected_point[2]
   '''
   pts_src = np.array([p1,p2,p4,p3], dtype="float32")
   pts_dst = np.array([[0, 0], [int((pts_src[1][0] - pts_src[0][0]) * 6), 0],
                       [int((pts_src[1][0] - pts_src[0][0]) * 6), int((pts_src[2][1] - pts_src[1][1]) * 4)],
                       [0, int((pts_src[2][1] - pts_src[1][1]) * 4)]], dtype="float32")

   h = cv2.getPerspectiveTransform(pts_src, pts_dst)
   print(h)
   mapped_width = 240*6
   mapped_height = 320*6
   mapped_im = cv2.warpPerspective(im, h, (mapped_width, mapped_height))
  # im_out = cv2.warpPerspective(img, h, (400, 100))
   img = cv2.polylines(img[0, :, :, :], np.int32([pts_src]), True, (123, 255, 255), 3)
   plt.subplot(1, 3, 1)
   plt.imshow(nn[0, :, :, 0])



   plt.subplot(1,3, 2)
   plt.imshow(img)


   plt.subplot(1, 3, 3)
   print(predected_gp_label.shape)
   plt.imshow(mapped_im)
   plt.show()
