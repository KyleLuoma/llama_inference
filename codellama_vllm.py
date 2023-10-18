"""
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
"""

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    max_tokens=1024,
    stop=[';', '#']
)

model = LLM(
    model='models/codellama/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/3773f63b4511b9e47a9a7ffc765eed7eb0169486',
    tokenizer='codellama/CodeLlama-7b-hf',
    dtype='float16',
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8
    )

prompt_file = open('./prompts/ntsb.txt')
prompt = prompt_file.read()
prompt_file.close()

output = model.generate(prompt, sampling_params)
print('SELECT', output[0].outputs[0].text[:-1].strip())