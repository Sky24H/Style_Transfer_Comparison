import os
import subprocess
from tqdm import tqdm
from neural_style import create_model, process_image
import time

with open('../selected_contents.txt', 'r') as r:
    selected_1 = r.read().splitlines()
with open('../selected_styles.txt', 'r') as r:
    selected_2 = r.read().splitlines()

save_dir = './results_test'
os.makedirs(save_dir, exist_ok=True)

cnn, layerList, gpu = create_model('models/vgg19-d01eb7cb.pth', 'max')

for content in selected_1[int(gpu):(int(gpu)+1)]:
    for style in tqdm(selected_2):
        start_time = time.time()
        save_name = os.path.basename(content)[:-4]+'_'+os.path.basename(style)[:-4]+'.png'
        # print(save_name)
        process_image(cnn, layerList, content, style, gpu=str(gpu))
        # cmd = 'CUDA_VISIBLE_DEVICES='+use_gpu+' python neural_style.py -style_image '+style+' -content_image '+content + ' -output_image .temp_'+use_gpu+'_.png'
        # subprocess.call(cmd, shell=True)
        # os.rename('.temp_'+use_gpu+'_.png', os.path.join(save_dir,save_name))
        print(time.time() - start_time)