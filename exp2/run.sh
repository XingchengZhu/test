# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1

python main.py --config=./exps/fcs/cifar100/5/first_stage-00.json
python main.py --config=./exps/fcs/cifar100/5/second_stage-00.json

python main.py --config=./exps/fcs/cifar100/5/first_stage-01.json
python main.py --config=./exps/fcs/cifar100/5/second_stage-01json

python main.py --config=./exps/fcs/cifar100/5/first_stage-10.json
python main.py --config=./exps/fcs/cifar100/5/second_stage-10.json

python main.py --config=./exps/fcs/cifar100/5/first_stage-11.json
python main.py --config=./exps/fcs/cifar100/5/second_stage-11.json