###############################
##  use small model to debug ##
###############################
model_name='qwen-v15-0.5b-chat'
vision_backbone="clip-vit-l-336px"
resize_strategy='resize-naive'
projector='avgpool2_8'
run_root_dir='debugs'
llm_max_length=2048

DEBUG_MODE='-m debugpy --listen 127.0.0.1:5678 --wait-for-client'


WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size 1 --run_root_dir $run_root_dir --model.llm_max_length $llm_max_length

# debug mode
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id python $DEBUG_MODE -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size 1 --run_root_dir $run_root_dir --model.llm_max_length $llm_max_length


#####################################
##  different vision/llm backbones ##
#####################################
gpu_id=0,1
gpu_num=2

# run settings
run_root_dir='./runs'  # saved ckpt dir
llm_max_length=1024

# model settings
model_name='phi2'  # vicuna-v15-7b or llama2-7b-pure or llama2-7b-chat or qwen-v15-0.5b or qwen-v15-1.8b ...
vision_backbone="dinosiglip-vit-so-384px"  # 'clip-vit-l-336px' 'siglip-vit-so400m-384px' 'siglip-vit-so400m' 'dinoclip-vit-l-336px'
resize_strategy='resize-naive'
projector='avgpool2_144' # 'maxpool2_144' 'cabstractor_144' 'gelu-mlp' 'qformer2_144'

# WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size 2 --run_root_dir $run_root_dir --model.llm_max_length $llm_max_length