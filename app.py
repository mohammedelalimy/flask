from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, render_template, request, jsonify

local_model_dir = "peft_model/"
HUGGING_FACE_USER_NAME = "elalimy"
model_name = "my_awesome_peft_finetuned_helsinki_model"
peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"
# Load model configuration (assuming it's saved locally)
config = PeftConfig.from_pretrained(local_model_dir, local_files_only=True)
# Load the base model from its local directory (replace with actual model type)cdd
model = AutoModelForSeq2SeqLM.from_pretrained(
    local_model_dir, return_dict=True, load_in_8bit=False)
# Load the tokenizer from its local directory (replace with actual tokenizer type)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
# # Load the Peft model (assuming it's a custom class or adaptation)
AI_model = PeftModel.from_pretrained(model, peft_model_id)


# Flask appclss
app = Flask(__name__, template_folder='templates')  # Specify the templates folder


def generate_translation(model, tokenizer, source_text, device="cpu"):
    # Encode the source text
    input_ids = tokenizer.encode(source_text, return_tensors='pt').to(device)

    # Move the model to the same device as input_ids
    model = model.to(device)

    # Generate the translation with adjusted decoding parameters
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=512,  # Adjust max_length if needed
        num_beams=4,
        length_penalty=5,  # Adjust length_penalty if needed
        no_repeat_ngram_size=4,
        early_stopping=True
    )

    # Decode the generated translation excluding special tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text to translate provided'}), 400

    text_to_translate = data['text']
    translated_text = generate_translation(AI_model, tokenizer, text_to_translate)

    return jsonify({'translated_text': translated_text})


if __name__ == "__main__":
    app.run()