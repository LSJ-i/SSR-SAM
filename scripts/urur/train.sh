now=$(date +"%Y%m%d_%H%M%S")
dataset='urur'
method='train_ssr_sam'
exp='sam'
split='0.16'

config=configs/urur_semi.yaml
labeled_id_path=datasets/URUR/split/$split/labeled.txt
unlabeled_id_path=datasets/URUR/split/$split/unlabeled.txt
val_path=datasets/URUR/split/val.txt
global_prototype_path=datasets/URUR/split/$split/global.pt
pretrain_path=exp/urur/train/sam/$split/best.pth
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --global_prototype_path $global_prototype_path\
    --pretrain_path $pretrain_path\
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --val-path $val_path\
    --log_dir $save_path --port $2 2>&1 | tee $save_path/$now.log