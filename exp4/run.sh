#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1

python main.py --config=./exps/fcs/cifar100/5/first_stage.json

# 定义超参数搜索空间
alphas=(0.1 1.0 2.0 10.0 20.0)
lambdas=(25 50 75 100)
kappas=(0.05 0.1 0.2 0.5)

# 基础配置文件路径
BASE_CONFIG="./exps/fcs/cifar100/5/second_stage.json"

for a in "${alphas[@]}"
do
    for l in "${lambdas[@]}"
    do
        for k in "${kappas[@]}"
        do
            # 构造唯一的标识名和临时配置文件名
            EXP_ID="alpha${a}_lambda${l}_kappa${k}"
            TMP_CONFIG="temp_config_${EXP_ID}.json"
            
            echo "------------------------------------------------"
            echo "正在运行实验: ${EXP_ID}"
            
            # 核心修复：Python 代码块必须顶格编写，防止 IndentationError
            python3 -c "
import json
with open('$BASE_CONFIG', 'r') as f:
    data = json.load(f)
data['alpha_mix'] = $a
data['lambda_mmd_base'] = $l
data['kappa'] = $k
data['log_name'] = 'exp4_${EXP_ID}'
with open('$TMP_CONFIG', 'w') as f:
    json.dump(data, f, indent=4)
"
            
            # 执行训练
            python main.py --config "$TMP_CONFIG"
            
            # 运行完删除临时配置文件
            rm "$TMP_CONFIG"
        done
    done
done