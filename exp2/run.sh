#!/bin/bash

# 1. 定义实验变量
seeds=(2001 2025)
rot_options=("true" "false")
mix_options=("true" "false")

# 基础配置文件根路径
BASE_DIR="./exps/fcs/cifar100"

for s in "${seeds[@]}"
do
    echo "================================================"
    echo "正在准备 Seed: ${s} 的共享 Task 0 权重"
    echo "================================================"

    # --- 准备 Stage 5 & 10 的共享 Task 0 (init_cls 50) ---
    S5_10_S1_LOG="shared_task0_50_seed${s}"
    S5_10_S1_JSON="temp_task0_50_s${s}.json"
    python3 -c "
import json
with open('${BASE_DIR}/5/first_stage.json', 'r') as f:
    data = json.load(f)
data['seed'] = [$s]
data['log_name'] = '${S5_10_S1_LOG}'
with open('${S5_10_S1_JSON}', 'w') as f:
    json.dump(data, f, indent=4)
"
    python main.py --config "$S5_10_S1_JSON"
    rm "$S5_10_S1_JSON"

    # --- 准备 Stage 20 的独立 Task 0 (init_cls 40) ---
    S20_S1_LOG="shared_task0_40_seed${s}"
    S20_S1_JSON="temp_task0_40_s${s}.json"
    python3 -c "
import json
with open('${BASE_DIR}/20/first_stage.json', 'r') as f:
    data = json.load(f)
data['seed'] = [$s]
data['log_name'] = '${S20_S1_LOG}'
with open('${S20_S1_JSON}', 'w') as f:
    json.dump(data, f, indent=4)
"
    python main.py --config "$S20_S1_JSON"
    rm "$S20_S1_JSON"

    # --- 开始消融实验循环 ---
    for r in "${rot_options[@]}"
    do
        for m in "${mix_options[@]}"
        do
            EXP_SUFFIX="seed${s}_rot${r}_mix${m}"
            
            # --- 执行 Stage 5 (50/10) ---
            echo ">>> 运行 Stage 5: ${EXP_SUFFIX}"
            S5_JSON="temp_S5_${EXP_SUFFIX}.json"
            S5_CKPT="logs/fcs/cifar100/50/10/${S5_10_S1_LOG}"
            python3 -c "
import json
with open('${BASE_DIR}/5/second_stage.json', 'r') as f:
    data = json.load(f)
data['seed'] = [$s]
data['enable_rot'] = json.loads('$r')
data['enable_mix'] = json.loads('$m')
data['ckpt_path'] = '${S5_CKPT}'
data['ckpt_num'] = 1
data['log_name'] = 'exp2_S5_${EXP_SUFFIX}'
with open('${S5_JSON}', 'w') as f:
    json.dump(data, f, indent=4)
"
            python main.py --config "$S5_JSON"
            rm "$S5_JSON"

            # --- 执行 Stage 10 (50/5) ---
            echo ">>> 运行 Stage 10: ${EXP_SUFFIX}"
            S10_JSON="temp_S10_${EXP_SUFFIX}.json"
            # 注意：此处权重目录中的 increment 需匹配 json 中的 5
            S10_CKPT="logs/fcs/cifar100/50/5/${S5_10_S1_LOG}"
            python3 -c "
import json
with open('${BASE_DIR}/10/second_stage.json', 'r') as f:
    data = json.load(f)
data['seed'] = [$s]
data['enable_rot'] = json.loads('$r')
data['enable_mix'] = json.loads('$m')
data['ckpt_path'] = '${S10_CKPT}'
data['ckpt_num'] = 1
data['log_name'] = 'exp2_S10_${EXP_SUFFIX}'
with open('${S10_JSON}', 'w') as f:
    json.dump(data, f, indent=4)
"
            python main.py --config "$S10_JSON"
            rm "$S10_JSON"

            # --- 执行 Stage 20 (40/3) ---
            echo ">>> 运行 Stage 20: ${EXP_SUFFIX}"
            S20_JSON="temp_S20_${EXP_SUFFIX}.json"
            S20_CKPT="logs/fcs/cifar100/40/3/${S20_S1_LOG}"
            python3 -c "
import json
with open('${BASE_DIR}/20/second_stage.json', 'r') as f:
    data = json.load(f)
data['seed'] = [$s]
data['enable_rot'] = json.loads('$r')
data['enable_mix'] = json.loads('$m')
data['ckpt_path'] = '${S20_CKPT}'
data['ckpt_num'] = 1
data['log_name'] = 'exp2_S20_${EXP_SUFFIX}'
with open('${S20_JSON}', 'w') as f:
    json.dump(data, f, indent=4)
"
            python main.py --config "$S20_JSON"
            rm "$S20_JSON"
        done
    done
done