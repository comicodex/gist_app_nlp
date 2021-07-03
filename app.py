from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import re
from heapq import nlargest

nlp = spacy.load("en_core_web_sm")


app = Flask(__name__)
app.config["DEBUG"] = True

def get_content(url):
    r = requests.get(url)
    text = r.text
    soup = BeautifulSoup(text, features="html.parser")
    all_p = soup.find_all(["p"])
    url_text = ""
    for p in all_p:
        url_text += p.text
    return url_text


def top_sentences(url):
    required_text = get_content(url)
    stopwords = list(STOP_WORDS)
    #punct = punctuation
    required_text = re.sub(r"[[0-9]*]", "", required_text)
    doc = nlp(required_text)

    # getting tokens
    words = [token.text for token in doc]

    # building dictionary of word frequencies to find out important words
    word_frequency = {}
    for word in doc:
        if word.text not in stopwords:
            if word.text not in punctuation:
                if word.text not in word_frequency.keys():
                    word_frequency[word.text] = 1
                else:
                    word_frequency[word.text] += 1

    # finding frequency of each word over most occurring word, normalizing word frequencies
    max_frequency = max(word_frequency.values())
    for word in word_frequency.keys():
        word_frequency[word] = word_frequency[word] / max_frequency

    # getting sentence tokens
    sentence_tokens = [sent for sent in doc.sents]

    # finding the most important sentences based on number of words (sentence score)
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequency.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequency[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequency[word.text.lower()]

    # using heapq library to find top 10 sentences with highest score
    summary = nlargest(8, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = " ".join(final_summary)
    return summary


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_gist", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        url = request.form.get("url")
        url_content = top_sentences(url)
        text = url_content
        original_text = get_content(url)
        length_original = len(original_text)
        length_summary = len(url_content)

        with open("static/files/summary.txt", "w", encoding="utf-8") as f:
            f.write(text)
            
        return render_template("summarize.html", response=text, length_source=length_original, length_summary=length_summary)

if __name__ == "__main__":
    app.run()
