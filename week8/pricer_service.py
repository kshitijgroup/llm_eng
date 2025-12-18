import modal
from modal import Image

# Setup - define our infrastructure with code!

app = modal.App("pricer_service")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "kshitijchaudhary"
RUN_NAME = "2025-12-16_14.53.09-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> str:

    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
    from peft import PeftModel
    import torch
    import re

    PREFIX= "Price is $"
    QUESTION = "What is the price to nearest dollar?"

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_quant_type="nf4")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, 
                                                      quantization_config=quant_config,
                                                      device_map="auto")

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(inputs["input_ids"], 
                                            attention_mask = inputs["attention_mask"],
                                                max_new_tokens=8)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    contents = result.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0