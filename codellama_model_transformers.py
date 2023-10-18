import torch
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import time
import os
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from device_maps import device_maps as d_mp

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'roundup_power2_divisions:[256:4,512:8,1024:16,>:32]'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

record_memory_history = False

if record_memory_history:
    torch.cuda.memory._record_memory_history()

#https://stackoverflow.com/questions/76872123/running-into-cuda-out-of-memory-when-running-llama2-13b-chat-model-on-multi-gpu

model_name = 'codellama/CodeLlama-7b-hf'
cache_dir = f'./models/{model_name}'

print(f'Loading tokenizer for {model_name}')
tokenizer = CodeLlamaTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    )
print(f'Loading model {model_name}')
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
    # device_map='balanced_low_0',
    device_map='auto',
    # device_map=d_mp._7bon16x,
    offload_state_dict = True,
    max_length=16000,
    # use_safetensors=False
    )
model = accelerator.prepare(model)

print(model.hf_device_map)

# model.to_bettertransformer()
# model.to('cuda')

mb_fig = 1000000

def print_mem_stats():
    for dev in range(0, 4):
            print(f"---- DEVICE {dev} ----")
            print("  Allocated:", torch.cuda.memory_allocated(dev)/mb_fig)
            print("  Max memory allocated:", torch.cuda.max_memory_allocated(dev)/mb_fig)
            print("  Memory reserved:", torch.cuda.max_memory_reserved(dev)/mb_fig)
            print("  Max memory reserved:", torch.cuda.max_memory_reserved(dev)/mb_fig)
            # print("  Stats:\n", torch.cuda.memory_stats(dev))

while True:

    print_mem_stats()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print_mem_stats()

    prompt = input('Enter a prompt:')
    if 'file' in prompt:
        filename = prompt.split(' ')[1]
        try:
            f = open(f'./prompts/{filename}.txt')
            prompt = f.read()
            f.close()
        except OSError as e:
            print(f"Unable to load {filename}.txt", e)
            continue


    print('Tokenizing')
    input_ids = tokenizer(
        prompt,
        return_tensors='pt'
        )['input_ids'].to('cuda')
    
    print("Prompt token size:", len(input_ids[0]))

    print('Running inference')
    s_time = time.perf_counter_ns()
    
    try:
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=256,
            num_return_sequences=1
            )
    except Exception as e:
        print(e)
        if record_memory_history:
            torch.cuda.memory._dump_snapshot("failure_memory_snapshot.pickle")

    output = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    query = 'SELECT ' + output.split(';')[0]
    stp_time = time.perf_counter_ns()
    print(query)
    print(f'elapsed time {stp_time - s_time}')
    input('Press enter to continue')
    if record_memory_history:
        torch.cuda.memory._dump_snapshot("success_memory_snapshot.pickle")