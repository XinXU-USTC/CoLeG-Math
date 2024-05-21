import os
import sys
import json
from prompt_generator import PromptGenerator
from openai_runner import query
from arguments import EntireArguments, DataArguments, RunningArguments
from data_processor import (
    load_gsm8k_dataset,
)
from transformers import HfArgumentParser
from omegaconf import OmegaConf
from api_keys import api_key

os.environ['OPENAI_API_KEY'] = api_key


def main():
    parser = HfArgumentParser([EntireArguments])
    if len(sys.argv) == 2:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        if sys.argv[1].endswith(".json") or sys.argv[1].endswith(".yaml"):
            arguments: EntireArguments = OmegaConf.load(os.path.abspath(sys.argv[1]))
        else:
            raise ValueError("Config file must be JSON or YAML file.")
    else:
        (arguments,) = parser.parse_args_into_dataclasses()
        arguments: EntireArguments = OmegaConf.structured(arguments)

    print(arguments)

    data_args = DataArguments.from_dict(arguments)
    running_args = RunningArguments.from_dict(arguments)
    dataset = load_gsm8k_dataset(data_args)

    if len(dataset) == 0:
        print("There is no data need to be run. Experiment finished.")
        return

    print(f"Model: {arguments.llm}, # Data Items: {len(dataset)}")

    prompt_generator = PromptGenerator(data_args, running_args)
    prompt_template = prompt_generator.prompt_template

    gen_kwargs = {
        "n": 1,
        "temperature": 0.0,
        "top_k": 0.7,
    }
    with open(arguments.output_filepath, "w", encoding="utf-8") as fout:
        for item in dataset:
            prompt = prompt_template.format(question=item['question'])
            response = query(prompt=prompt, model='gpt-3.5-turbo-0125', **gen_kwargs)
            resp_item = {
                "id": item['id'],
                "response": response,
                "prompt": prompt,
            }
            fout.write(json.dumps(resp_item) + '\n')


if __name__ == "__main__":
    main()
