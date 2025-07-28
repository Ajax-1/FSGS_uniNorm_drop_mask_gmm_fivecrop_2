#!/usr/bin
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# data=( "flower" "horns"  "fortress" "room" "trex" "fern" "orchids" "leaves")
# data=("trex")
unseen_num=360
mvs_config=./mvs_modules/configs/config_mvsformer.json
res=8
n_blocks=4
output=uniNorm_mvp_0605
#output=FSGS_0603_rUni_on_vdpt
export CUDA_VISIBLE_DEVICES=4
#for ((i=1; i<=3; i++)); do
for dataset in "${data[@]}"; do
		    python train.py --source_path "../dataset/dataset/LLFF/$dataset" --model_path "output/$output/$dataset" --eval --n_views 3 --sample_pseudo_interval 1 -r $res --dataset LLFF \
			--stage train   --mvs_config $mvs_config --total_virtual_num $unseen_num --n_blocks $n_blocks
		    python render.py --source_path "../dataset/dataset/LLFF/$dataset"  --model_path  "output/$output/$dataset" --iteration 10000 -r $res  
		    python metrics.py --source_path "../dataset/dataset/LLFF/$dataset"  --model_path  "output/$output/$dataset" --iteration 10000
 			
           
        done	
for dataset in "${data[@]}"; do
		    python metrics.py --source_path "../dataset/dataset/LLFF/$dataset"  --model_path  "output/$output/$dataset" --iteration 10000 
        done	
# export CUDA_VISIBLE_DEVICES=7
# #for ((i=1; i<=3; i++)); do
# for dataset in "${data[@]}"; do
# 		    python train.py --source_path "../dataset/dataset/LLFF/$dataset" --model_path "output/output_FSGS_0502/$dataset" --eval --n_views 3 --sample_pseudo_interval 1 -r 8 
# 		    # python render.py --source_path "../dataset/dataset/LLFF/$dataset"  --model_path  "output/output_FSGS_0430/$dataset" --iteration 10000 -r 8
# 			# python metrics.py --source_path "../dataset/dataset/LLFF/$dataset"  --model_path  "output/output_FSGS_0430/$dataset" --iteration 10000
 			
           
#     done	  

