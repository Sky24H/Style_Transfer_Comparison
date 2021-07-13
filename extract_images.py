import os
import glob
from tqdm import tqdm
import shutil


all_images = glob.glob(os.path.join('./2_AdaIN/results_test', '*'))
save_root = './images_from_nst/2'
domains = ['monet', 'ink', 'water', 'vangogh']
for d in domains:
    os.makedirs(os.path.join(save_root, d), exist_ok=True)
for path in tqdm(all_images):
    domain = os.path.basename(path).split('_')[0]
    save_path = os.path.join(save_root, domain,os.path.basename(path))
    shutil.copyfile(path, save_path)
