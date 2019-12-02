import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

backgrounds_folder_path = r"C:\Users\erez.posner\PycharmProjects\open3d_test\backgrounds"
objects_folder_path = r"C:\Users\erez.posner\PycharmProjects\open3d_test\objects"

backgrounds_files = glob.glob(backgrounds_folder_path + '\*')
objects_files = glob.glob(objects_folder_path + '\*')
count =0
for i in range(2):
    for background in backgrounds_files:
        for object in objects_files:

            back = cv2.imread(background, -1)
            object = cv2.imread(object, -1)


            object[object[..., 3] == 0] = 0
            img_mask = object.copy()
    # img_mask[img_mask[...,3]>0]=255
            img_mask = img_mask[:, :, :3]
            object = object[:, :, :3]

            mask = np.zeros(back.shape)

            position = (
            np.random.randint(0, back.shape[0] - img_mask.shape[0]), np.random.randint(0, back.shape[1] - img_mask.shape[1]))

            mask[position[0]:position[0] + img_mask.shape[0], position[1]:position[1] + img_mask.shape[1], :] = object

            mask_bin = mask.copy()
            mask_bin[mask_bin > 0] = 255
            mask_bin = mask_bin[:, :, 0]

            plt.figure()
            plt.imshow(mask.astype(np.uint8))

            plt.figure()
            plt.imshow(mask_bin)
            plt.figure()

            plt.imshow(back)
            output = np.zeros(back.shape)

            output[mask_bin == 0] = back[mask_bin == 0]
            output[mask_bin == 255] = mask[mask_bin == 255]
            plt.imshow(output.astype(np.uint8))
            # plt.show()
            plt.close('all')
            cv2.imwrite('output\{:03d}.png'.format(count),output)
            count = count+1
