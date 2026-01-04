from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./lora_sama"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = """### Instruction:
Explain ICAAP as per SAMA guidelines.

### Input:
ICAAP is a comprehensive process through which banks assess their capital adequacy.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=80)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
