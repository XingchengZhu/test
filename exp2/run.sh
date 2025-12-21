#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1

python main.py --config=./exps/fcs/cifar100/5/first_stage.json

# 定义消融开关
rot_options=(true false)
mix_options=(true false)


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
        
        # 动态修改 JSON
        # 注意：Python 中的 True/False 需要转为首字母大写的布尔值
        python3 -c "
            import json
            with open('$BASE_CONFIG', 'r') as f:
                data = json.load(f)
            data['enable_rot'] = $r
            data['enable_mix'] = $m
            data['log_name'] = 'exp2_ablation_${EXP_ID}'
            with open('$TMP_CONFIG', 'w') as f:
                json.dump(data, f, indent=4)
            "
        
        python main.py --config "$TMP_CONFIG"
        
        rm "$TMP_CONFIG"
    done
done

