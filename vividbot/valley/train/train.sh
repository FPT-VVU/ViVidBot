NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node 6 --nnodes 1 --node_rank=0 --master_addr 127.0.0.1 --master_port 6006 vividbot/valley/train/train.py --conf $1