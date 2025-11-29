python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=10086 \
    zero_test.py \
    --config configs/levir_test.yaml \
    --test_path datasets/LEVIR/test.txt