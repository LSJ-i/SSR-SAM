now=$(date +"%Y%m%d_%H%M%S")
dataset='inria'
method='train_s3sam_alpha'
split='0.16'
sample_num=16

config=configs/inria_semi.yaml

alpha=0.5
beta=1.0
gamma=0.5
exp="${alpha}_${beta}_${gamma}"
labeled_id_path=datasets/Aerial/split/$split/labeled.txt
unlabeled_id_path=datasets/Aerial/split/$split/unlabeled.txt
val_path=datasets/Aerial/split/val.txt
global_prototype_path=datasets/Aerial/split/$split/global.pt
pretrain_path=exp/inria/_with_lora/sam/$split/best.pth
save_path=exp/$dataset/$method/$split/$exp

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --global_prototype_path $global_prototype_path\
    --pretrain_path $pretrain_path\
    --sample_num $sample_num\
    --alpha $alpha\
    --beta $beta\
    --gamma $gamma\
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --val-path $val_path\
    --log_dir $save_path --port $2 2>&1 | tee $save_path/$now.log


now=$(date +"%Y%m%d_%H%M%S")
dataset='inria'
method='train_s3sam_alpha'
split='0.16'
sample_num=16

config=configs/inria_semi.yaml

alpha=1.0
beta=0.5
gamma=0.5
exp="${alpha}_${beta}_${gamma}"
labeled_id_path=datasets/Aerial/split/$split/labeled.txt
unlabeled_id_path=datasets/Aerial/split/$split/unlabeled.txt
val_path=datasets/Aerial/split/val.txt
global_prototype_path=datasets/Aerial/split/$split/global.pt
pretrain_path=exp/inria/_with_lora/sam/$split/best.pth
save_path=exp/$dataset/$method/$split/$exp/$sample_num

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --global_prototype_path $global_prototype_path\
    --pretrain_path $pretrain_path\
    --sample_num $sample_num\
    --alpha $alpha\
    --beta $beta\
    --gamma $gamma\
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --val-path $val_path\
    --log_dir $save_path --port $2 2>&1 | tee $save_path/$now.log

now=$(date +"%Y%m%d_%H%M%S")
dataset='inria'
method='train_s3sam_alpha'
split='0.16'
sample_num=16

config=configs/inria_semi.yaml

alpha=0.5
beta=0.5
gamma=0.5
exp="${alpha}_${beta}_${gamma}"
labeled_id_path=datasets/Aerial/split/$split/labeled.txt
unlabeled_id_path=datasets/Aerial/split/$split/unlabeled.txt
val_path=datasets/Aerial/split/val.txt
global_prototype_path=datasets/Aerial/split/$split/global.pt
pretrain_path=exp/inria/_with_lora/sam/$split/best.pth
save_path=exp/$dataset/$method/$split/$exp/$sample_num

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --global_prototype_path $global_prototype_path\
    --pretrain_path $pretrain_path\
    --sample_num $sample_num\
    --alpha $alpha\
    --beta $beta\
    --gamma $gamma\
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --val-path $val_path\
    --log_dir $save_path --port $2 2>&1 | tee $save_path/$now.log


