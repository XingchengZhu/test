# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1

python main.py --config=./exps/fcs/cifar100/5/first_stage.json # base stage
python main.py --config=./exps/fcs/cifar100/5/second_stage.json # incremental learning