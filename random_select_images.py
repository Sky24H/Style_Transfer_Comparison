import glob
import random
import os


def select(data_root, filename, how_many):
    if isinstance(data_root, list):
        selected_images = []
        for dir_ in data_root:
            images = glob.glob(os.path.join(dir_, '*'))
            random.shuffle(images)
            images = random.sample(images, how_many)
            selected_images += images
    else:
        all_images = glob.glob(os.path.join(data_root, '*'))
        random.shuffle(all_images)
        selected_images = random.sample(all_images, how_many)

    with open(filename, 'w') as w:
        for path in selected_images:
            w.write(path)
            w.write('\n')
    return selected_images


data_root_1 = '/mnt/data/huang/datasets/synthesized_image'
data_root_2 = ['/mnt/data/huang/research_workspace/playground/real_resized_512/real_water', '/mnt/data/huang/research_workspace/playground/real_resized_512/real_ink',
                '/mnt/data/huang/research_workspace/playground/real_resized_512/real_monet', '/mnt/data/huang/research_workspace/playground/real_resized_512/real_vangogh']
filename_1 = './selected_contents.txt'
filename_2 = './selected_styles.txt'

selected_1 = select(data_root_1, filename_1, how_many=50)
selected_2 = select(data_root_2, filename_2, how_many=50)


print(len(selected_1), len(selected_2))