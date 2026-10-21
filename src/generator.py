from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_model(model_name=MODEL_NAME):
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto")
    return llm_tokenizer, model


def build_prompt(query, content):
    return (
        "Answer the question using ONLY information "
        "explicitly stated in the context. "
        "If the answer is not present in the context, "
        "respond exactly: I don't know."
        f"Context:\n{content}\n\n"
        f"Question: {query}"
    )


def get_answer_text(prompt, llm_tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False
    )
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    answer_text = llm_tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return answer_text
