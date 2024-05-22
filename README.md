<h1 align="center"><span style="color: #3498db;">CoLeG-Math</h1>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![E-GSM License](https://img.shields.io/badge/E--GSM%20License-MIT-yellow.svg)](https://lbesson.mit-license.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

- Official code repository for paper "Can LLMs Solve Longer Math Word Problems Better?"
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
bash scripts/xxx
```
or
```bash
python xx
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
coming soon
```
