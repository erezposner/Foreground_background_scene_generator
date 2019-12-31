import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from pathlib import Path
import os
import re
backgrounds_folder_path = r"D:\Repositories\MaDE\Output\Processed\background\Components\RGB"
objects_folder_path = r"D:\Repositories\MaDE\Output\Processed\2019-12-25-11.40.55_joined\Components\RGB"
output_folder_final = Path("D:\Repositories\Foreground_background_scene_generator\output4")
# backgrounds_files = glob.glob(backgrounds_folder_path + '\*')
objects_files = glob.glob(objects_folder_path + '\*_01.png')
name = str(Path(objects_folder_path).parent.parent.name)
s = 2
streams = np.zeros(50)
for i in range(1):
    for object in objects_files:
        # fgbg = cv2.createBackgroundSubtractorMOG2()
        try:
            stream_id = int(re.split('stream|_',Path(object).stem)[1])
            object_folders = os.listdir(objects_folder_path)

            # object_folders = list(filter(os.path.isdir,object_folders))
            count = int(streams[stream_id])
            # for object in object_folders:
            # object_masks_path = (Path(objects_folder_path) / object / 'Components' / 'Masks')
            # object_rgb_path = (Path(objects_folder_path) / object / 'Components' / 'RGB')
            # if object_masks_path.exists():
            #     masks = glob.glob(str(object_masks_path / 'stream{:03d}*'.format(stream_id)))
            #     for mask in masks:
            #         try:
            if count ==0:
                o = str(Path(object).parent.parent.parent.parent / 'background' / 'Components' / 'RGB' / Path(object).name)

                rgb_image = cv2.imread(o,-1)
            else:
                rgb_image = cv2.imread(object,-1)

            mask_image = np.zeros(rgb_image.shape)
            # print('background - {} , object - {}'.format(background, object))

            output_folder_name = '{}'.format(name) +   Path(object).stem.split('_')[0]
            output_folder = output_folder_final / str(output_folder_name)
            output_folder.mkdir(parents=True, exist_ok=True)
            in_folder = output_folder / 'input'
            gt_folder = output_folder / 'groundtruth'
            in_folder.mkdir(parents=True, exist_ok=True)
            gt_folder.mkdir(parents=True, exist_ok=True)
            # back = cv2.imread(background, -1)
            # indices = np.where(mask_image)
            # output = back.copy()
            # output[indices[0], indices[1]] = rgb_image[indices[0], indices[1]]

            # plt.figure()
            # plt.imshow(mask_image)
            plt.figure()

            plt.imshow(rgb_image)
            # plt.figure()
            #
            # plt.imshow(output)
            plt.close('all')
            # if count > 1:
            cv2.imwrite(str(output_folder / in_folder / 'in{:06d}.png'.format(count)), rgb_image)
            # cv2.imwrite(str(output_folder / gt_folder / 'gt{:06d}.png'.format(count)), mask_image)
            cv2.imwrite(str(output_folder / gt_folder / 'gt{:06d}.png'.format(count)), mask_image)
            # else:
            #     cv2.imwrite(str(output_folder / in_folder / 'in{:06d}.png'.format(count)), back)
            #     cv2.imwrite(str(output_folder / gt_folder / 'gt{:06d}.png'.format(count)),
            #                 np.zeros(mask_image.shape))
            streams[stream_id] = streams[stream_id] + 1
        except:
            print('err')
            pass
print('done')
