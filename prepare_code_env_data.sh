# donwload code
git clone https://github.com/yaolinli/prismatic-vlms.git
git checkout rsh


# set conda envs
conda create -n prism-vlms python=3.10
conda activate prism-vlms
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

cd prismatic-vlms
pip install -e .
pip install packaging ninja
ninja --version; echo $?
pip install flash-attn --no-build-isolation
pip install open-clip-torch==2.23.0
pip install datasets


# download data
DATA_DIR="./data"
python scripts/preprocess.py --dataset_id "llava-v1.5-instruct" --root_dir $DATA_DIR


# test run code
gpu_id=0,1,2,3
gpu_num=4
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
  --model.llm_backbone_id $model_name  --run_id "one-stage_"${model_name}"+"${projector} --model.finetune_per_device_batch_size 1
