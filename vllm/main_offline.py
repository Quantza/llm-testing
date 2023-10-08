from vllm import LLM, SamplingParams

g_prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def prompt_llm(llm, prompts, sampling_params):
    # Source: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-125m")

    # temperature: Float to control the randomness of sampling (0=deterministic, greedy; 1=random)
    # top_p: Float that controls cumulative probability of the top tokens to consider (1=all tokens)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    prompt_llm(llm, g_prompts, sampling_params)


if __name__ == "__main__":
    main()
else:
    print ("Imported")