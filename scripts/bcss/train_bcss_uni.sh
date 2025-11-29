
now=$(date +"%Y%m%d_%H%M%S")
dataset='bcss'
method='train_s3sam_uni_wo_lora'
exp='sam'
split='853' #['106','213','426','853']

config=configs/bcss_semi_uni.yaml
labeled_id_path=datasets/BCSS/split/$split/labeled.txt
unlabeled_id_path=datasets/BCSS/split/$split/unlabeled.txt
val_path=datasets/BCSS/split/val.txt
global_prototype_path=datasets/BCSS/split/$split/global.pt
save_path=exp/$dataset/$method/zero

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --global_prototype_path $global_prototype_path\
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --val-path $val_path\
    --log_dir $save_path --port $2 2>&1 | tee $save_path/$now.log

