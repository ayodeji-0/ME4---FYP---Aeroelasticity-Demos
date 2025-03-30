from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("chatbot/models/t5_model")
tokenizer = T5Tokenizer.from_pretrained("chatbot/models/t5_model")

# def summarize_text(text):
#     input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
#     output_ids = model.generate(input_ids, max_length=150, min_length=30, num_beams=4)
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def summarize_text(context, question=None):
    if question:
        prompt = f"answer the question: {question} using: {context}"
    else:
        prompt = f"summarize: {context}"
        
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(
        input_ids, max_length=150, min_length=40, 
        num_beams=4, length_penalty=2.0, early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
