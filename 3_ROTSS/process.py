import os
import subprocess
from tqdm import tqdm
import time
from main import process_image, create_model
import torch
import sys
args = sys.argv

use_gpu = args[4]

with open('../selected_contents.txt', 'r') as r:
    selected_1 = r.read().splitlines()
with open('../selected_styles.txt', 'r') as r:
    selected_2 = r.read().splitlines()
len_ = len(selected_1)//8

save_dir = './results_test'
os.makedirs(save_dir, exist_ok=True)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
vgg_encoder, pyramid, select_tensor, indices_generator = create_model(device)

for content in tqdm(selected_1[int(use_gpu)*len_:(int(use_gpu)+1)*len_]):
    for style in tqdm(selected_2):
        start_time = time.time()
        style_name = style.split('/')[-2].split('_')[-1]
        # print(style_name)
        save_name = style_name+'_'+os.path.basename(content)[:-4]+'_'+os.path.basename(style)[:-4]+'.png'
        process_image(vgg_encoder, pyramid, select_tensor, indices_generator, content, style, '.temp_'+use_gpu+'_.png', device)
        # cmd = 'CUDA_VISIBLE_DEVICES='+use_gpu+' python main.py '+content+' '+style+' --output .temp_'+use_gpu+'_.png'
        # cmd = 'CUDA_VISIBLE_DEVICES='+use_gpu+' python neural_style.py -style_image '+style+' -content_image '+content + ' -output_image .temp_'+use_gpu+'_.png'
        # print(cmd)
        # subprocess.call(cmd, shell=True)
        os.rename('.temp_'+use_gpu+'_.png', os.path.join(save_dir,save_name))
        # print(time.time() - start_time)
