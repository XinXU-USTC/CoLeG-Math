import argparse
import json
import re
import jsonlines
from fraction import Fraction
from torch.utils.data import Dataset
from vllm import LLM, SamplingParams
from transformers.modeling_utils import PreTrainedModel
import os
import sys
import torch
import numpy as np

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split("The answer is: ")
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            if "/" in match.group():
                denominator = match.group().split("/")[1]
                numerator = match.group().split("/")[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == "0":
                        return round(float(numerator.replace(",", "")))
                    else:
                        frac = Fraction(match.group().replace(",", ""))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(",", "")) == float("inf"):
                    return None
                return round(float(match.group().replace(",", "")))
        else:
            return None
    else:
        return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


class PureTextDataset(Dataset):
    def __init__(self, instances: list, tokenizer):
        super().__init__()
        self.instances = instances
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def collate_fn(self, batch):
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")


def readjsonl2list(name):
    data = []  # Create an empty list to store the dictionaries

    with open(name, "r") as file:
        for line in file:
            dict_obj = json.loads(line)
            data.append(dict_obj)
    return data


def gsm8k_test(
    args,
    model,
    start=0,
    end=MAX_INT,
    batch_size=1,
    tensor_parallel_size=1,
):
    # set output directory
    output_filename = args.out_filepath

    # INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    gsm8k_ids = []
    gsm8k_questions = []

    if args.prompt_instruction:
        problem_prompt = args.prompt_instruction
    elif model == "deepseek-math-7b-rl" or model == "deepseek-math-7b-instruct":
        problem_prompt = "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
    else:
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
    print(f"prompt ===== {problem_prompt}")
    test_set = readjsonl2list(args.dataset_filepath)
    dataset = test_set

    if os.path.isfile(output_filename):
        exist_data = readjsonl2list(output_filename)
        exist_data_ids = set([item["id"] for item in exist_data])
        data_remains = [item for item in dataset if item["id"] not in exist_data_ids]
        dataset = data_remains

    if len(dataset) == 0:
        print("There is no data need to be run. Experiment finished.")
        return

    for data in dataset:
        temp_instr = problem_prompt.format(instruction=data["question"])
        gsm8k_ins.append(temp_instr)
        temp_ans = data["answer"]
        gsm8k_answers.append(temp_ans)
        temp_id = data["id"]
        gsm8k_ids.append(temp_id)
        gsm8k_questions.append(data["question"])

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print(f"length ==== {len(gsm8k_ins)}")
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = [
        "Question:",
        "Question",
        "USER:",
        "USER",
        "ASSISTANT:",
        "ASSISTANT",
        "Instruction:",
        "Instruction",
        "Response:",
        "Response",
    ]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens
    )
    print(f"sampling ===== {sampling_params}")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
    )
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    gsm8k_preds = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        zip(gsm8k_ins, res_completions, gsm8k_answers)
    ):
        y_pred = extract_answer_number(completion)
        gsm8k_preds.append(y_pred)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer.replace(",", "")))
        else:
            result.append(False)
            temp = {"question": prompt, "output": completion, "answer": prompt_answer}
            invalid_outputs.append(temp)
    acc_ = sum(result) / len(result)
    print(
        f"len invalid outputs ==== {len(invalid_outputs)}, invalid_outputs=== {invalid_outputs}"
    )
    print(f"start === {start}, end ==== {end}")
    print(f"gsm8k length ==== {len(result)}, acc ==== {acc_}")
    store = []
    for q, id, completion, prompt_answer, ans, acc in zip(
        gsm8k_questions, gsm8k_ids, res_completions, gsm8k_answers, gsm8k_preds, result
    ):
        temp = {
            "question": q,
            "id": id,
            "acc": acc,
            "response": completion,
            "answer": ans,
            "ground-truth": prompt_answer,
        }
        store.append(temp)
    with open(output_filename, "w", encoding="utf-8") as fout:
        for item in store:
            fout.write(json.dumps(store) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=8
    )  # tensor_parallel_size
    parser.add_argument("--number_extraction", type=bool, default=True)
    parser.add_argument(
        "--number_extraction_llm", type=str, default="gpt-3.5-turbo-0125"
    )
    parser.add_argument("--dataset_filepath", type=str, default="")
    # no need to change
    parser.add_argument("--prompt_type", type=str, default="zero-shot-cot")
    parser.add_argument("--prompt_round", type=int, default=-1)
    parser.add_argument("--prompt_instruction", type=str, default="")
    parser.add_argument("--output_filepath", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.isfile(os.path.join(args.output_filepath)):
        dataset = readjsonl2list(args.output_filepath)
    else:
        gsm8k_test(
            args,
            model=args.model,
            start=args.start,
            end=args.end,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    
    # answer extraction
    if args.number_extraction:
        output_filename = os.path.join(os.path.abspath(os.path.dirname(args.output_filepath)), "numerical_result.jsonl")
        cmd = "cd ..\n"
        cmd += "cd proprietary-llms\n"
        cmd += f"python main.py --llm {args.number_extraction_llm} --dataset_filepath {os.path.abspath(args.dataset_filepath)} --output_filepath {output_filename} --is_number_extraction True"
        os.system(cmd)
        results = readjsonl2list(output_filename)
        extract_acc_list = [not item["extract_failed"] for item in results]
        print(f"Extraction Accuracy: {np.mean(extract_acc_list)}")
        acc_list = [item["acc"] for item in results]
        print(f"Acc after extraction: {np.mean(acc_list)}")
