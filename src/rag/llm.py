import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


def get_hf_llm(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.2,
    max_new_tokens: int = 450,
    top_p: float = 0.75,
    **kwargs,
):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=top_p,
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)
    return llm
