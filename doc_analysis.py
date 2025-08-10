from transformers import pipeline
import stanza
from langdetect import detect
import re
import json
import os
from keybert import KeyBERT

stanza.download('en')  # once

nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')
kw_model = KeyBERT()
sentiment_pipeline = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)


'''def extract_metadata(text):
    metadata = {}

    try:
        metadata['language'] = detect(text)
    except:
        metadata['language'] = "unknown"

    first_line = text.strip().split("\n")[0]
    metadata['title'] = first_line if len(first_line) < 120 else first_line[:120] + "..."

    date_match = re.search(r"\b\d{1,2}[-/ ]\d{1,2}[-/ ]\d{2,4}\b", text)
    metadata['date'] = date_match.group(0) if date_match else None

    return metadata


def extract_keywords(text, top_n=10):
    doc = nlp(text)
    keywords = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in ("NOUN", "PROPN"):
                keywords.append(word.text)

    keywords = list(dict.fromkeys(keywords))
    return keywords[:top_n]


def summarize_text(text, max_words=200):
    if len(text.split()) < 50:
        return text
    summary = summarizer(text, max_length=max_words, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.entities:
        entities.setdefault(ent.type, set()).add(ent.text)
    return {k: list(v) for k, v in entities.items()}


def analyze_document(text, output_path="data/doc_analysis.json"):
    metadata = extract_metadata(text)
    keywords = extract_keywords(text)
    summary = summarize_text(text)
    entities = extract_entities(text)

    analysis = {
        "metadata": metadata,
        "keywords": keywords,
        "summary": summary,
        "entities": entities
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    return analysis'''


def analyze_document(text, file_path=None):
    analysis = {}

    # Metadata
    if file_path:
        file_stats = os.stat(file_path)
        analysis["metadata"] = {
            "filename": os.path.basename(file_path),
            "size_kb": round(file_stats.st_size / 1024, 2)
        }

    # Keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english")
    analysis["keywords"] = [kw[0] for kw in keywords]

    # Named Entities using Stanza
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.type} for sentence in doc.sentences for ent in sentence.ents]
    analysis["entities"] = entities

    # Sentiment
    sentiment = sentiment_pipeline(text[:512])  # Truncate to model limit
    analysis["sentiment"] = sentiment[0]

    # Summary
    summary = summarizer(text[:1024], max_length=100, min_length=30, do_sample=False)
    analysis["summary"] = summary[0]["summary_text"]

    return analysis
