from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = "Qwen/Qwen3-0.6B"
model_name = r"D:\Models\Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = """### Instruction:
Explain ICAAP as per SAMA guidelines.

### Input:
ICAAP is a comprehensive process through which banks assess their capital adequacy.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=80)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
