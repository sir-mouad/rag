from transformers import AutoModelForCausalLM, AutoTokenizer
import re

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_model():
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto")
    return llm_tokenizer, model


def build_prompt(query, content):
    return (
        # "Answer the question using only the following context. "
        # # "If the answer is not in the context, say you don't know.\n\n"
        "Answer the question using ONLY information explicitly stated in the context. "
        # "If the context does not explicitly contain the answer, respond exactly with "
        # "\"I don't know.\" Do not infer, guess, or use general knowledge.\n\n"
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
    # answer_text = re.sub(r"<think>.*?</think>\s*", "",
    #                      answer_text, flags=re.DOTALL).strip()
    return answer_text
