#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1

python main.py --config=./exps/fcs/cifar100/5/first_stage.json

# 定义消融开关
rot_options=("true" "false")
mix_options=("true" "false")

BASE_CONFIG="./exps/fcs/cifar100/5/second_stage.json"

for r in "${rot_options[@]}"
do
    for m in "${mix_options[@]}"
    do
        # 构造实验标识
        EXP_ID="rot_${r}_mix_${m}"
        TMP_CONFIG="temp_ablation_${EXP_ID}.json"
        
        echo "------------------------------------------------"
        echo "正在运行消融实验: ${EXP_ID}"
        
        # 核心修复：
        # 1. 使用 python3 -c 时，内部代码左对齐，防止 IndentationError
        # 2. 使用 json.loads('$r') 将字符串 "true"/"false" 转换为 Python 的 True/False
        # 3. 使用 'w' 模式打开文件，防止重复运行导致的 JSON 格式损坏
        python3 -c "
import json
with open('$BASE_CONFIG', 'r') as f:
    data = json.load(f)
data['enable_rot'] = json.loads('$r')
data['enable_mix'] = json.loads('$m')
data['log_name'] = 'exp2_ablation_${EXP_ID}'
with open('$TMP_CONFIG', 'w') as f:
    json.dump(data, f, indent=4)
"
        
        # 运行实验
        python main.py --config "$TMP_CONFIG"
        
        # 运行完删除临时配置文件
        rm "$TMP_CONFIG"
    done
done

