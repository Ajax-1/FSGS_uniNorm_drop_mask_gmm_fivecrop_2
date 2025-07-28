## LLFF
#!/usr/bin
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# data=( "fern" "flower"  "fortress" "horns" "leaves" "room" "trex" "orchids")

data=("trex" )
unseen_num=360
mvs_config=./mvs_modules/configs/config_mvsformer.json
export CUDA_VISIBLE_DEVICES=5
output=LLFF_huatu_test
res=8
##四个块权重为1.0，中间块多0.05。
ncc_weight=1.0
unseen_v_ncc_weight=0.05
#0表示不使用fivecrop
crop_flag=1
for dataset in "${data[@]}"; do
        # source_path=../dataset/dataset/LLFF/$dataset
        python train_uni.py  --source_path ../dataset/dataset/LLFF/$dataset --model_path output/$output/$dataset --eval  --n_views 3 --sample_pseudo_interval 1  -r $res \
        --dataset LLFF 	--stage train   --mvs_config $mvs_config --total_virtual_num $unseen_num  --unseen_v_ncc_weight $unseen_v_ncc_weight --ncc_weight $ncc_weight \
        --crop_flag $crop_flag
        python render.py --source_path ../dataset/dataset/LLFF/$dataset  --model_path  output/$output/$dataset --iteration 10000 -r $res --render_depth
        python metrics.py --source_path ../dataset/dataset/LLFF/$dataset  --model_path  output/$output/$dataset --iteration 10000
    done

# for dataset in "${data[@]}"; do

#         python metrics.py --source_path ../dataset/dataset/LLFF/$dataset  --model_path  output/$output/$dataset --iteration 10000
#     done
