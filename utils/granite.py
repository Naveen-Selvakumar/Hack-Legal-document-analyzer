# utils/granite.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "ibm-granite/granite-3.1-1b-a400m-instruct"

@st.cache_resource(show_spinner=False)
def load_granite_pipeline():
    import importlib
    device = 0 if torch.cuda.is_available() else -1
    model_kwargs = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    # Try to use 8bit quantization if bitsandbytes and GPU are available
    try:
        bitsandbytes_spec = importlib.util.find_spec("bitsandbytes")
        if bitsandbytes_spec is not None and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return pipe

def generate_with_granite(prompt: str, max_new_tokens: int = 128, do_sample: bool = False):
    """
    Generate text using IBM Granite-3.1-1b-a400m-instruct model.
    Args:
        prompt (str): Input text prompt.
        max_new_tokens (int): Maximum tokens to generate (default 128, max 256).
        do_sample (bool): Whether to use sampling (default: False for deterministic output).
    Returns:
        str: Generated text from the model.
    """
    granite_generator = load_granite_pipeline()
    # Clamp max_new_tokens for safety
    max_new_tokens = max(32, min(max_new_tokens, 256))
    result = granite_generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return result[0]["generated_text"]

def chunk_text(text, max_words=100):
    """Split text into chunks by paragraph, then by word count for large docs."""
    import re
    paras = re.split(r'\n{2,}', text)
    chunks = []
    for para in paras:
        words = para.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i+max_words])
            if chunk.strip():
                chunks.append(chunk.strip())
    return chunks
