import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from pathlib import Path
import os
import re
backgrounds_folder_path = r"C:\Erez.Posner_to_NAS\Foreground_background_scene_generator\backgrounds"
objects_folder_path = r"\\fs01\Algo\ML\Datasets\NeuralPCL\Data"
output_folder_final = Path("C:\Erez.Posner_to_NAS\Foreground_background_scene_generator\output2")
backgrounds_files = glob.glob(backgrounds_folder_path + '\*')
objects_files = glob.glob(objects_folder_path + '\*')

s = 2
for i in range(4):
    for background in backgrounds_files:
        # fgbg = cv2.createBackgroundSubtractorMOG2()

        stream_id = int(re.split('stream|_',Path(background).stem)[1])
        object_folders = os.listdir(objects_folder_path)

        # object_folders = list(filter(os.path.isdir,object_folders))
        count = 1
        for object in object_folders:
            object_masks_path = (Path(objects_folder_path) / object / 'Components' / 'Masks')
            object_rgb_path = (Path(objects_folder_path) / object / 'Components' / 'RGB')
            if object_masks_path.exists():
                masks = glob.glob(str(object_masks_path / 'stream{:03d}*'.format(stream_id)))
                for mask in masks:
                    try:
                        rgb_image = cv2.imread(str(object_rgb_path / Path(mask).stem.replace('MASK', 'RGB')) + '.png',
                                               -1)
                        mask_image = cv2.imread(mask, -1)
                        # print('background - {} , object - {}'.format(background, object))

                        output_folder_name = Path(background).stem.split('_')[0]
                        output_folder = output_folder_final / str(output_folder_name)
                        output_folder.mkdir(parents=True, exist_ok=True)
                        in_folder = output_folder / 'input'
                        gt_folder = output_folder / 'groundtruth'
                        in_folder.mkdir(parents=True, exist_ok=True)
                        gt_folder.mkdir(parents=True, exist_ok=True)
                        back = cv2.imread(background, -1)
                        indices = np.where(mask_image)
                        output = back.copy()
                        output[indices[0], indices[1]] = rgb_image[indices[0], indices[1]]

                        plt.figure()
                        plt.imshow(mask_image)
                        plt.figure()

                        plt.imshow(back)
                        plt.figure()

                        plt.imshow(output)
                        plt.close('all')
                        if count > 1:
                            cv2.imwrite(str(output_folder / in_folder / 'in{:06d}.png'.format(count)), output)
                            cv2.imwrite(str(output_folder / gt_folder / 'gt{:06d}.png'.format(count)), mask_image)
                            cv2.imwrite(str(output_folder / gt_folder / 'gt{:06d}.png'.format(count)), mask_image)
                        else:
                            cv2.imwrite(str(output_folder / in_folder / 'in{:06d}.png'.format(count)), back)
                            cv2.imwrite(str(output_folder / gt_folder / 'gt{:06d}.png'.format(count)),
                                        np.zeros(mask_image.shape))
                        count = count + 1
                    except:
                        print('err')
                        pass
print('done')
