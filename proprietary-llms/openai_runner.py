import openai


def query(prompt: str, model: str = "gpt-3.5-turbo-0125", **gen_kwargs):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful asistant."},
            {"role": "user", "content": prompt},
        ],
        **gen_kwargs,
    )
    if model != "gpt-3.5-turbo-instruct":
        result = [
            response["choices"][i]["message"]["content"]
            for i in range(len(response["choices"]))
        ]
    else:
        result = [
            response["choices"][i]["text"] for i in range(len(response["choices"]))
        ]
    return result


if __name__ == "__main__":
    pass
