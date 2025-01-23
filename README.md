<h1 align="center"><span style="color: #3498db;">CoLeG-Math</h1>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![E-GSM License](https://img.shields.io/badge/E--GSM%20License-MIT-yellow.svg)](https://lbesson.mit-license.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

- Official code repository for paper [Can LLMs Solve Longer Math Word Problems Better?](https://arxiv.org/abs/2405.14804)
- **E**xtended **G**rade-**S**chool **M**ath (**E-GSM**) benchmark is an arithmetic reasoning dataset built upon GSM8K, by extending problem descriptions into longer ones.
- E-GSM is constructed to evaluate **Co**ntext **Le**ngth **G**eneralizability (**CoLeG**) of LLMs, the ability of LLMs to solve long math word problems.
- For proprietary LLMs, we introduce **Co**ndition-**Re**trieving Instruction (CoRe), an instructional prompt.
- For opensource LLMs, we suggest incorporating **extension** as an auxiliary task of finetuning and release our SFT data.

## Quick Start
Clone CoLeG-Math and install the required packages:

```bash
git clone xxx
cd CoLeG-Math
pip install -r requirements.txt
```
For vLLM installation problems, please refer to [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html).


## Dataset Usage

For now, E-GSM and our SFT data are under `./data` forder. Huggingface Link: coming soon...

## Evaluation on E-GSM
For proprietary LLMs, you need to put your key in `proprietary-llms/api_keys.py`
```bash
cd proprietary-llms
python3 main.py config.yaml
```
or
```bash
cd proprietary-llms
bash ../scripts/eval_proprietary.sh
```
or
```bash
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
```
For opensource LLMs:
```bash
cd opensource-llms
bash ../scripts/eval_opensource.sh
```
or
```bash
python opensource-llms/eval_gsm8k.py --model "path/to/save" --dataset_filepath data/E-GSM/Q1.jsonl --output_filepath Q1_results.jsonl
```
## Training
You need to prepare the LLM to be fine-tuned
```bash
bash scripts/train.sh
```

Thanks for the open source code of [MetaMath](https://github.com/meta-math/MetaMath/), [WizardMath](https://github.com/nlpxucan/WizardLM/tree/main/WizardMath) and [RFT](https://github.com/OFA-Sys/gsm8k-ScRel/tree/main). Some of our codes are based on them.

## Citation

Please cite our paper if you use our dataset or extend our work:
```bibtex
@article{xu2024coleg-math,
  title={Can LLMs Solve longer Math Word Problems Better?},
  author={Xu, Xin and Xiao, Tong and Chao, Zitong and Huang, Zhenya and Yang, Can and Wang, Yang},
  journal={arXiv preprint arXiv:2405.14804},
  year={2024}
}
```
