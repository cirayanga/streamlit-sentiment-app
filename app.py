import transformers
import nltk
nltk.download('punkt')

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Removed invalid line
transformers
import pandas as pd
nltk
from rake_nltk import Rake
import plotly

