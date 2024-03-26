
# train only stage 2
QNUM=32 
echo $QNUM 
# Run from the root of the repository
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "one-stage+7b_qformer2_${QNUM}"  --run_id "qformer2_${QNUM}"  --model.arch_specifier "qformer2_${QNUM}"  --model.finetune_per_device_batch_size 1


# train two stages


QNUM=64 
echo $QNUM 
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" --stage "align" \
  --model.model_id "one-stage+7b_qformer2_${QNUM}"  --run_id "stage1_qformer2_${QNUM}"  --model.arch_specifier "qformer2_${QNUM}"

# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" --stage "finetune"  --pretrained_checkpoint "/home/lilei/prismatic-vlms/runs/llava-qformer/stage1_qformer2_${QNUM}/checkpoints/latest-checkpoint.pt" \
  --model.model_id "one-stage+7b_qformer2_${QNUM}"  --run_id "s1_s2_qformer2_${QNUM}"  --model.arch_specifier "qformer2_${QNUM}"