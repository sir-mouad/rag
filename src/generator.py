from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME =  "Qwen/Qwen3-0.6B"

def load_model():
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return llm_tokenizer, model


def build_prompt(query, content):
    return (
        "Answer the question using only the following context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{content}\n\n"
        f"Question: {query}"
    )