import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_language_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Load in Bits & Bytes configuration
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the pretrained model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        quantization_config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the padding token to the eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str)-> str:
    # Create a chat history
    chat = [
        {"role": "system", 
         "content": "You are a nutrition tracker assistant. You are helping a user track their daily food intake. Based on the food items the user has consumed, you are providing a summary of their nutrition intake."},
        {"role": "user", "content": prompt},
    ]

    # Tokenize the query
    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").cuda()

    # Generate the response
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=256)

    # Decode the response
    output = tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True)

    response  = output.split("[/INST] ")[0]

    return response


if __name__ == "__main__":
    model, tokenizer = load_language_model()

    prompt = "Caesar Salad, Grilled Chicken, Hot Dog, Candy."

    response = generate_response(model, tokenizer, prompt)

    print(response)