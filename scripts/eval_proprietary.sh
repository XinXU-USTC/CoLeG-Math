#!/bin/bash

python3 main.py \
    --llm gpt-3.5-turbo-0125 \
    --n 1 \
    --top_p 0.7 \
    --temperature 0.0 \
    --max_tokens 1024 \
    --prompt_name zero-shot-cot \
    --generate_log_file \
    --use_core_instruction \
    --dataset_filepath /path/to/datafile \
    --output_filepath /path/to/save
