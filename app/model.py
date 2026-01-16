from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
import torch
import gc


model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Use TinyLlama instead of Mistral (1.1B parameters vs 7B)
#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#model_id="microsoft/Phi-3-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4-bit quantization - uses only ~4GB instead of 14GB
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # Compute in float16 for speed
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    max_length=2048,
    temperature=0.1,
    do_sample=False,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

print(f"✓ Model loaded")
print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
