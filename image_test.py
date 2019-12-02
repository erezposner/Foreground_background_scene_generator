import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from pathlib import Path
backgrounds_folder_path = r"C:\Erez.Posner_to_NAS\Foreground_background_scene_generator\backgrounds"
objects_folder_path = r"C:\Erez.Posner_to_NAS\Foreground_background_scene_generator\objects"
output_folder_final = Path("C:\Erez.Posner_to_NAS\Foreground_background_scene_generator\output")
backgrounds_files = glob.glob(backgrounds_folder_path + '\*')
objects_files = glob.glob(objects_folder_path + '\*')
count = 0
s = 2
for i in range(4):
    for background in backgrounds_files:
        for object in objects_files:
            try:
                output_folder_name = Path(background).stem
                output_folder = output_folder_final / str(output_folder_name)
                output_folder.mkdir(parents=True,exist_ok = True)
                in_folder = output_folder / 'input'
                gt_folder = output_folder / 'groundtruth'
                in_folder.mkdir(parents=True,exist_ok = True)
                gt_folder.mkdir(parents=True,exist_ok = True)
                back = cv2.imread(background, -1)
                dim = (back.shape[1] // s, back.shape[0] // s)
                back = cv2.resize(back, dim, interpolation=cv2.INTER_AREA)
                object = cv2.imread(object, -1)
                sca = np.random.uniform(2,4)

                dim = (int(object.shape[1] // sca), int(object.shape[0] // sca))
                object = cv2.resize(object, dim, interpolation=cv2.INTER_AREA)

                object[object[..., 3] == 0] = 0
                img_mask = object.copy()
                # img_mask[img_mask[...,3]>0]=255
                img_mask = img_mask[:, :, :3]
                object = object[:, :, :3]

                mask = np.zeros(back.shape)

                position = (
                    np.random.randint(0, back.shape[0] - img_mask.shape[0]),
                    np.random.randint(0, back.shape[1] - img_mask.shape[1]))

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
                cv2.imwrite(str(output_folder / in_folder /  'in{:05d}.png'.format(count)), output)
                cv2.imwrite(str(output_folder / gt_folder / 'gt{:05d}.png'.format(count)), mask_bin)
                count = count + 1
            except:
                print('err')
                pass
