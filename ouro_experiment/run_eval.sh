#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_PATH="/ssd/yangxw/LoopedModel/ouro_experiment/outputs/ouro-sft-reg-randomloop-lastmathtime-400K/epoch0_checkpoint" 

OUTPUT_PATH="./eval_outputs/math500.json"

echo "Starting evaluation for Ouro model: ${MODEL_PATH}"
echo "Results will be saved to: ${OUTPUT_PATH}"

HF_ALLOW_CODE_EVAL="1" lm_eval --model vllm \
    --model_args '{
        "pretrained": "'"${MODEL_PATH}"'",
        "trust_remote_code": true,
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
        "enforce_eager": true,
        "hf_overrides": {
            "total_ut_steps": 9
        }
    }' \
    --tasks minerva_math500 \
    --batch_size auto \
    --num_fewshot 0 \
    --apply_chat_template True \
    --log_samples \
    --confirm_run_unsafe_code \
    --system_instruction "You are a helpful assistant that can assist users with reasoning." \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs '{
        "max_gen_toks": 2048
    }' 

echo "Evaluation finished."
#--include_path XXXX \ for including additional test sets if needed
