from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

app = FastAPI()


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
session = ort.InferenceSession("model/model.onnx")

class PromptInput(BaseModel):
    prompt: str
    max_tokens: int = 30

@app.post("/generate")
def generate_text(data: PromptInput):
    prompt = data.prompt
    max_tokens = data.max_tokens
    k = 10 


    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.copy()

    for _ in range(max_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

        outputs = session.run(None, {
            "input_ids": generated_ids,
            "attention_mask": np.ones_like(generated_ids),
            "position_ids": position_ids
        })

        logits = outputs[0][0, -1]

        top_k_ids = logits.argsort()[-k:][::-1]
        top_k_logits = logits[top_k_ids]
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        next_token_id = int(np.random.choice(top_k_ids, p=probs))

        next_token = np.array([[next_token_id]])
        generated_ids = np.concatenate([generated_ids, next_token], axis=1)

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return {"text": output_text}