gpu_id=0,1
gpu_num=2
model_name='phi2'
vision_backbone="dinosiglip-vit-so-384px"
resize_strategy='resize-naive'
projector='avgpool2_144'

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size 2