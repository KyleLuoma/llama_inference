python -m vllm.entrypoints.api_server \
    --model models/codellama/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/3773f63b4511b9e47a9a7ffc765eed7eb0169486 \
    --dtype float16 \
    --tokenizer codellama/CodeLlama-7b-hf \
    --tensor-parallel-size 4