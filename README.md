# 🏦 Fine-Tuned Small Language Model for SAMA & Basel (Financial Services)

Domain-specific regulatory assistant fine-tuned on SAMA ICAAP guidelines using LoRA

## Overview

This project demonstrates end-to-end fine-tuning of a small open-source language model on financial services regulatory data.

Instead of calling large APIs, I fine-tuned Qwen3-0.6B on SAMA ICAAP & Basel risk concepts to improve:

Regulatory tone

Instruction adherence

Domain-specific explanations

Built under CPU-only constraints, mirroring real enterprise prototyping.

# 🎯 Business Use Case

Internal AI assistant for banking consultants and risk teams

Capabilities:

Explain ICAAP & Basel frameworks

Classify banking risks

Respond in formal, regulator-aligned language

## 🛠 Tech Stack

Model: Qwen3-0.6B

Fine-tuning: LoRA (PEFT)

Framework: Hugging Face Transformers

Compute: CPU only

Data: Instruction-tuned JSONL from SAMA document

## 🧪 Dataset Design

Training samples were manually derived from SAMA ICAAP guidelines.

Each regulatory paragraph was converted into:

Concept explanations

Risk classifications

Consultant-style rewrites

Example:

{
  "instruction": "Explain ICAAP as per SAMA guidelines.",
  "input": "ICAAP is a comprehensive internal capital assessment process.",
  "output": "ICAAP is a SAMA-mandated process through which banks assess capital adequacy relative to their risk profile and strategy."
}

# 🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Test Base Model
python test_base.py

3️⃣ Fine-Tune with LoRA
python finetune_lora.py

4️⃣ Test Fine-Tuned Model
python test_finetuned.py
