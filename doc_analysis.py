import os
import json
import re
from langdetect import detect
import stanza
from keybert import KeyBERT
from transformers import pipeline
import torch

# stanza.download('en')  # once

# nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner')
# kw_model = KeyBERT()
# sentiment_pipeline = pipeline("sentiment-analysis")
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)


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


# Second edit - added sentiment analysis
'''def analyze_document(text, file_path=None):
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

    return analysis'''


# Third Edit
stanza.download("en", processors="tokenize,pos,lemma,ner", verbose=False)
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,ner", use_gpu=torch.cuda.is_available())

kw_model = KeyBERT()

device = 0 if torch.cuda.is_available() else -1

_summary_pipeline = None
_sentiment_pipeline = None


def get_summary_pipeline():
    global _summary_pipeline
    if _summary_pipeline is None:
        _summary_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    return _summary_pipeline


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis", device=device)
    return _sentiment_pipeline


def extract_metadata(text, file_path=None):
    metadata = {}
    # language
    try:
        metadata["language"] = detect(text)
    except:
        metadata["language"] = "unknown"

    # title (first non-empty line)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    metadata["title"] = lines[0][:200] if lines else "Unknown"

    # date detection (simple heuristics)
    date_match = re.search(r"\b\d{1,2}[-/ ]\d{1,2}[-/ ]\d{2,4}\b", text)
    metadata["date"] = date_match.group(0) if date_match else None

    if file_path:
        try:
            st = os.stat(file_path)
            metadata["filename"] = os.path.basename(file_path)
            metadata["size_kb"] = round(st.st_size / 1024, 2)
        except:
            pass
    return metadata


def extract_keywords(text, top_n=10):
    try:
        kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
        return [k[0] for k in kws]
    except Exception:
        # fallback: noun phrases using POS tagging
        doc = nlp(text)
        noun_phrases = []
        for sent in doc.sentences:
            np = []
            for word in sent.words:
                if word.upos in ["ADJ", "NOUN", "PROPN"]:
                    np.append(word.text)
                else:
                    if np:
                        noun_phrases.append(" ".join(np))
                        np = []
            if np:
                noun_phrases.append(" ".join(np))

        seen = []
        for c in noun_phrases:
            if c not in seen:
                seen.append(c)
            if len(seen) >= top_n:
                break
        return seen


def extract_entities(text):
    doc = nlp(text)
    ents = {}
    for sent in doc.sentences:
        for ent in sent.ents:
            ents.setdefault(ent.type, set()).add(ent.text)
    return {k: list(v) for k, v in ents.items()}


def summarize_text(text, max_length=120):
    if len(text.split()) < 40:
        return text
    summarizer = get_summary_pipeline()
    short = " ".join(text.split()[:2000])
    try:
        out = summarizer(short, max_length=max_length, min_length=20, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        return short[:500] + "..."


def sentiment(text):
    s = get_sentiment_pipeline()
    try:
        return s(text[:512])[0]
    except Exception:
        return {"label": "NEUTRAL", "score": 0.0}


def analyze_document(text, file_path=None, save_json=True, out_path="data/doc_analysis.json"):
    analysis = {}
    analysis["metadata"] = extract_metadata(text, file_path=file_path)
    analysis["keywords"] = extract_keywords(text)
    analysis["entities"] = extract_entities(text)
    analysis["summary"] = summarize_text(text)
    analysis["sentiment"] = sentiment(text)

    if save_json:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    return analysis
