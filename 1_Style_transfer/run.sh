CUDA_VISIBLE_DEVICES=0 python process.py -gpu 0 -output_image .temp_0_.png &
CUDA_VISIBLE_DEVICES=1 python process.py -gpu 0 -output_image .temp_1_.png &
CUDA_VISIBLE_DEVICES=2 python process.py -gpu 0 -output_image .temp_2_.png &
CUDA_VISIBLE_DEVICES=3 python process.py -gpu 0 -output_image .temp_3_.png &
CUDA_VISIBLE_DEVICES=4 python process.py -gpu 0 -output_image .temp_4_.png &
CUDA_VISIBLE_DEVICES=5 python process.py -gpu 0 -output_image .temp_5_.png &
CUDA_VISIBLE_DEVICES=6 python process.py -gpu 0 -output_image .temp_6_.png &
CUDA_VISIBLE_DEVICES=7 python process.py -gpu 0 -output_image .temp_7_.png