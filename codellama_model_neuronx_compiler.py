#https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/bert-base-cased-finetuned-mrpc-inference-on-trn1-tutorial.html
# Use aws_neuron_venv_pytorch environment

import torch
import torch_neuronx
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import transformers

def encode(tokenizer, *inputs, max_length=16000, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
        # torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
    )

model_name = 'codellama/CodeLlama-7b-hf'
cache_dir = f'./models/{model_name}'

print(f'Loading tokenizer for {model_name}')
tokenizer = CodeLlamaTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16
    )
tokenizer.pad_token = tokenizer.eos_token
print(f'Loading model {model_name}')

model = LlamaForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.float16,
    cache_dir=cache_dir,
    max_length=8000,
    use_safetensors=False,
    torchscript=True
    )

test_prompt = "#A SQL query to show the number of cars by color\nSELECT"

paraphrase = encode(tokenizer, test_prompt)

# Compile the model for Neuron
print("Compiling Model")
model_neuron = torch_neuronx.trace(
    model, 
    paraphrase,
    compiler_workdir='./compiler_wd'
    )

# Save the TorchScript for inference deployment
filename = './models/neuron/CodeLlama-7b-hf.pt'
torch.jit.save(model_neuron, filename)