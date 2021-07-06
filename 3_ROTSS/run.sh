CUDA_VISIBLE_DEVICES=0 python process.py -gpu 0 -num 0 &
CUDA_VISIBLE_DEVICES=1 python process.py -gpu 0 -num 1 &
CUDA_VISIBLE_DEVICES=2 python process.py -gpu 0 -num 2 &
CUDA_VISIBLE_DEVICES=3 python process.py -gpu 0 -num 3 &
CUDA_VISIBLE_DEVICES=4 python process.py -gpu 0 -num 4 &
CUDA_VISIBLE_DEVICES=5 python process.py -gpu 0 -num 5 &
CUDA_VISIBLE_DEVICES=6 python process.py -gpu 0 -num 6 &
CUDA_VISIBLE_DEVICES=7 python process.py -gpu 0 -num 7