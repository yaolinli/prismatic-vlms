

QNUM=$1 
echo $QNUM 
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "one-stage+7b_qformer2_${QNUM}"  --run_id "qformer2_${QNUM}"  --model.arch_specifier "qformer2_${QNUM}"
