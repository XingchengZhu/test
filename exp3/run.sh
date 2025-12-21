#!/bin/bash

# 定义要测试的模型列表，加入 vit_base
vit_models=("vit_tiny" "vit_small" "vit_base")

FIRST_BASE="./exps/fcs/cifar100/5/first_stage.json"
SECOND_BASE="./exps/fcs/cifar100/5/second_stage.json"

for m in "${vit_models[@]}"
do
    echo "================================================"
    echo "开始执行模型实验: ${m}"
    echo "================================================"

    # --- 步骤 1: 第一阶段 ---
    LOG_FIRST="${m}_first_stage"
    TMP_FIRST="temp_first_${m}.json"
    
    python3 -c "
import json
with open('$FIRST_BASE', 'r') as f:
    data = json.load(f)
data['convnet_type'] = '$m'
data['log_name'] = '$LOG_FIRST'
# 如果是 vit_base，调小 batch_size 防止显存溢出
if '$m' == 'vit_base':
    data['batch_size'] = 8
with open('$TMP_FIRST', 'w') as f:
    json.dump(data, f, indent=4)
"
    python main.py --config "$TMP_FIRST"


    # --- 步骤 2: 增量阶段 ---
    LOG_SECOND="${m}_second_stage"
    TMP_SECOND="temp_second_${m}.json"
    CKPT_DIR="logs/fcs/cifar100/50/10/$LOG_FIRST"

    python3 -c "
import json
with open('$SECOND_BASE', 'r') as f:
    data = json.load(f)
data['convnet_type'] = '$m'
data['log_name'] = '$LOG_SECOND'
data['ckpt_path'] = '$CKPT_DIR'
data['ckpt_num'] = 1
if '$m' == 'vit_base':
    data['batch_size'] = 8
with open('$TMP_SECOND', 'w') as f:
    json.dump(data, f, indent=4)
"
    python main.py --config "$TMP_SECOND"

    rm "$TMP_FIRST" "$TMP_SECOND"
done