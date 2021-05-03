from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import LongformerTokenizer, EncoderDecoderModel

app = Flask(__name__)


# Load model and tokenizer
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

def get_content(url):
    request_url = requests.get(url)
    text = request_url.text
    soup = BeautifulSoup(text, "html.parser")
    all_p = soup.find_all("p")
    url_text = ""
    for p in all_p:
        url_text += p.text
    return url_text


def summarize(url):
    required_text = get_content(url)

    # Tokenize and summarize
    input_ids = tokenizer(required_text, return_tensors="pt", max_length=2000, truncate=True).input_ids
    output_ids = model.generate(input_ids, max_length=2000)

    # Get the summary from the output tokens
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        url_content = summarize(url)
        return url_content
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)