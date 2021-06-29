import glob
import random
import os


def select(data_root, filename, how_many):
    all_images = glob.glob(os.path.join(data_root, '*'))
    random.shuffle(all_images)

    selected_images = random.sample(all_images, how_many)
    with open(filename, 'w') as w:
        for path in selected_images:
            w.write(path)
            w.write('\n')
    return selected_images


data_root_1 = '/mnt/data/huang/datasets/4in1_datasets_artworks/train_img'
data_root_2 = '/mnt/data/huang/datasets/4in1_datasets_artworks/train_img'
filename_1 = './selected_contents.txt'
filename_2 = './selected_styles.txt'

selected_1 = select(data_root_1, filename_1, how_many=10)
selected_2 = select(data_root_2, filename_2, how_many=100)


print(len(selected_1), len(selected_2))