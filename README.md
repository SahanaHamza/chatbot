from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load free chatbot model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("✅ Chatbot ready! Type 'quit' to exit.\n")

chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        break

    inputs = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt")

    if chat_history_ids is not None:
        inputs = torch.cat([chat_history_ids, inputs], dim=-1)

    chat_history_ids = model.generate(
        inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    reply = tokenizer.decode(
        chat_history_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", reply)
