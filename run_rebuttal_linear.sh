# gpu settings
gpu_id=0,1
gpu_num=2
run_root_dir='runs'
finetune_per_device_batch_size=2


#####################################
##    E1: siglip 224 | 256->144     #
#####################################
model_name='phi2'
vision_backbone='siglip-vit-so400m'
resize_strategy='resize-naive'
projector='gelu-mlp'

CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size $finetune_per_device_batch_size --run_root_dir $run_root_dir --model.llm_max_length 1024


#####################################
##    E3: siglip 384 | 729->144     #
#####################################
model_name='phi2'
vision_backbone='siglip-vit-so400m-384px'
resize_strategy='resize-naive'
projector='gelu-mlp'

CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size $finetune_per_device_batch_size --run_root_dir $run_root_dir --model.llm_max_length 1024


  #####################################
##    E2: clip 336 | 576->144     #
#####################################
model_name='phi2'
vision_backbone='clip-vit-l-336px'
resize_strategy='resize-naive'
projector='gelu-mlp'

CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size $finetune_per_device_batch_size --run_root_dir $run_root_dir --model.llm_max_length 1024