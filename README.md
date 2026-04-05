# 🚀 DeepSeek From Scratch
A comprehensive **from-scratch implementation of modern LLMs components inspired by DeepSeek**, it covers attention variants, KV cache optimization, Rotary Positional Encoding (RoPE), Mixture of Experts (MoE), and inference-focused improvements used in scalable LLM systems.

This repository covers multiple advanced transformer optimizations including:
- Multi-Head Attention (MHA)
- KV Cache optimization
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Multi-Head Latent Attention (MLA)
- Rotary Positional Encoding (RoPE)
- Mixture of Experts (MoE)


This project is inspired by concepts from the *“DeepSeek from Scratch”* book, implemented independently to deepen practical understanding.


## 📌 Project Motivation

Large Language Models like DeepSeek introduce several optimizations over standard transformers to improve:
- ⚡ Inference speed  
- 💾 Memory efficiency  
- 📈 Scalability  

This project is an effort to **understand and implement these techniques from scratch** without relying on high-level libraries, built for learning, experimentation, and research purposes.

---

## 🧠 Implemented Components

This project focuses on building core transformer components and optimizations from scratch, commonly used in modern large language models.

### 🔹 Attention Mechanisms
- Multi-Head Attention (MHA)
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Multi-Head Latent Attention (MLA)

### 🔹 Inference Optimizations
- KV Cache for efficient autoregressive decoding
- Reduced redundant computation during generation

### 🔹 Positional Encoding
- Rotary Positional Encoding (RoPE)

### 🔹 Advanced Architectures
- Mixture of Experts (MoE)

### 🔹 Experimental Notebooks
- Step-by-step exploration of attention outputs
- KV cache behavior analysis
- Token-level decoding experiments

---

## Future Improvements

- Integration of quantization techniques to further optimize model efficiency (to be added).
- Development of a complete training pipeline, including training and evaluation loops.
- Building an end-to-end integrated pipeline combining all implemented components.
