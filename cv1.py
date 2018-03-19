from __future__ import absolute_import, division, print_function
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import multiprocessing
import os
import re
import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory

nltk.download("punkt")
nltk.download("stopwords")

UPLOAD_FOLDER = '/home/harsha/Downloads/codefd/files'
ALLOWED_EXTENSIONS = set(['pdf','txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    book_filenames = sorted(glob.glob("files/*.txt"))

    corpus_raw = u""
    for book_filename in book_filenames:
        print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        print("Corpus is now {0} characters long".format(len(corpus_raw)))
        print()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus_raw)

    def sentence_to_wordlist(raw):
        clean = re.sub("[^a-zA-Z]"," ", raw)
        words = clean.split()
        return words
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))	

    num_features = 300
    min_word_count = 3
    context_size = 7
    downsampling = 0.5*1e-5            		
    num_workers = multiprocessing.cpu_count()
    seed = 1
    maha2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )
    maha2vec.build_vocab(sentences)
    maha2vec.train(sentences,total_examples=maha2vec.corpus_count, epochs=maha2vec.iter)
    if not os.path.exists("trained"):
        os.makedirs("trained")
    maha2vec.save(os.path.join("trained", "maha2vec.w2v"))		
    return  render_template('upload.html')

@app.route('/display', methods=['POST'])
def test():
    maha2vec = w2v.Word2Vec.load(os.path.join("trained", "maha2vec.w2v"))
        
    similarities = maha2vec.most_similar_cosmul(
        positive=[request.form['word3'], request.form['word1']],
        negative=[request.form['word2']]
    )
    start2 = similarities[0][0]
    return render_template('display.html',name=start2)

if __name__ == "__main__":
	app.run(
        host="0.0.0.0",
        port=int("5000"),
        debug=True
    )