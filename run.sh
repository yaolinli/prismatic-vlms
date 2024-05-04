# gpu settings
gpu_id=0,1
gpu_num=2
run_root_dir='runs'

# test code
model_name='phi2'
vision_backbone="dinosiglip-vit-so-384px"
resize_strategy='resize-naive'
projector='avgpool2_144'

# WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.arch_specifier 'no-align+'$projector \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size 2 --run_root_dir $run_root_dir --model.llm_max_length 1024


#####################################
##    E1: vision backbones exps     #
#####################################
model_name='phi2'
resize_strategy='resize-naive'
projectors=('avgpool2_144' 'cabstractor_144')
vision_backbones=('clip-vit-l-336px' 'siglip-vit-so400m-384px')


for vision_backbone in "${vision_backbones[@]}"
do
    for projector in "${projectors[@]}"
    do
        # WANDB_MODE=offline \
        CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
        --model.type "reproduction-llava-v15+7b" \
        --model.model_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} \
        --model.vision_backbone_id $vision_backbone \
        --model.image_resize_strategy $resize_strategy \
        --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} --model.arch_specifier 'no-align+'$projector --model.finetune_per_device_batch_size 2 --run_root_dir $run_root_dir --model.llm_max_length 1024
    done
done

#################################
##    E2: llava-v15-7B exps     #
#################################
# 1) make sure model exists: /home/v-shuhuairen/mycontainer/ckpt/official_ckpts/vicuna-7b-v1.5


# e1: clip-vit-l-336px + vicuna-v15-7b + 2*gelu-mlp
model_name="vicuna-v15-7b"
vision_backbone="clip-vit-l-336px" 
resize_strategy='resize-naive'
projector='gelu-mlp'

# WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} --model.arch_specifier 'no-align+'$projector --model.finetune_per_device_batch_size 1 --run_root_dir $run_root_dir

# e2: clip-vit-l-336px + vicuna-v15-7b + avgpool2_144
model_name="vicuna-v15-7b"
vision_backbone="clip-vit-l-336px" 
resize_strategy='resize-naive'
projector='avgpool2_144'

# WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} --model.arch_specifier 'no-align+'$projector --model.finetune_per_device_batch_size 1 --run_root_dir $run_root_dir

# e3: dinosiglip-vit-so-384px + vicuna-v15-7b + avgpool2_144
model_name="vicuna-v15-7b"
vision_backbone="dinosiglip-vit-so-384px" 
resize_strategy='resize-naive'
projector='avgpool2_144'

# WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} --model.arch_specifier 'no-align+'$projector --model.finetune_per_device_batch_size 1 --run_root_dir $run_root_dir

# e4: dinosiglip-vit-so-384px + vicuna-v15-7b + cabstractor2_144
model_name="vicuna-v15-7b"
vision_backbone="dinosiglip-vit-so-384px" 
resize_strategy='resize-naive'
projector='cabstractor_144'

# WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --standalone --nnodes 1 --nproc-per-node $gpu_num scripts/pretrain.py \
  --model.type "reproduction-llava-v15+7b" \
  --model.model_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} \
  --model.vision_backbone_id $vision_backbone \
  --model.image_resize_strategy $resize_strategy \
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${vision_backbone}"+"${projector} --model.arch_specifier 'no-align+'$projector --model.finetune_per_device_batch_size 1 --run_root_dir $run_root_dir
