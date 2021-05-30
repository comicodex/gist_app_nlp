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
    page = requests.get(url).text
    soup = BeautifulSoup(page, "html.parser")
    headline = soup.find("h1").get_text()
    p_tags = soup.find_all("p")
    # Get the text from each of the “p” tags and strip surrounding whitespace.
    p_tags_text = [tag.get_text().strip() for tag in p_tags]
    # Filter out sentences that contain newline characters '\n'
    sentence_list = [sentence for sentence in p_tags_text if not '\n' in sentence]
    sentence_list = [sentence for sentence in sentence_list if "." in sentence]
    article = " ".join(sentence_list)
    return article


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
    #final_summary = [word.text for word in summary]
    #summary = " ".join(final_summary)
    return summary


@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/upload_gist", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        url = request.form.get("url")
        url_content = top_sentences(url)
        original_text = get_content(url)
        length_original = len(original_text)
        length_summary = len(url_content)
        return render_template("summarize2.html", response=url_content, length_source=length_original, length_summary=length_summary)

if __name__ == "__main__":
    app.run(debug=True)