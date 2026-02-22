#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import html
from nltk.corpus import stopwords, words
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np
from scipy.special import softmax
from catboost import CatBoostClassifier
from textblob import TextBlob


NER_COLS = ["ORG", "GPE", "NORP", "DATE", "CARDINAL", "PRODUCT", "ORDINAL", "LOC", "LAW"]
BASE_COLS = ["keyword", "country"]
LABEL_COL = "PCL_category"
TRAIN_TEXT_MIN_LEN = 3
IMPORTANCE = [3,2,1,2,5]
CATEGORICAL_COLS = ["keyword", "country"] 
RANDOM_SEED = 42

TRAIN_DATA_PATH = "data/PCL_train_dataset.tsv"
VAL_DATA_PATH = "data/PCL_val_dataset.tsv"
TEST_DATA_PATH = "data/PCL_test_dataset.tsv"

LABEL_COL = "PCL_category"
 
class FeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, nlp, stop_words, vocab):
        self.nlp = nlp
        self.ner_cols = NER_COLS
        self.base_cols = BASE_COLS
        self.stop_words = stop_words
        self.vocab = vocab
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        
        texts = X["text"].astype(str).tolist()
        results = []
        
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=32)):
 
            row_features = {
                # Constructed features
                **self._clean_text(doc),
                **self._punctuation_features(doc),
                **self._ner_counts(doc),
                **self._verb_noun_ratio(doc),
                # **self._count_misspellings(doc),
                # Base columns
                "keyword": X.iloc[i]["keyword"],
                "country": X.iloc[i]["country"]
            }
            
            
            results.append(row_features)
        
        return pd.DataFrame(results)
    
    def _clean_text(self, doc):
        text = doc.text
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\t", " ")).strip()
        text = re.sub(r"[^\w\s]", "", text)
        words = text.lower().split()
        tokens = [w for w in words if w not in self.stop_words]
        return {"cleaned_text": " ".join(tokens)}
    
    def _count_misspellings(self, doc):
        misspelled = 0
        tokens = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]
        
        for word in tokens:
            if word not in self.vocab:
                misspelled += 1

        return {"misspelled_ratio": misspelled / len(doc) if len(doc) > 0 else 0}

    def _punctuation_features(self, doc):
        text = doc.text
        sentence_len = max(len(text.split()), 1)
        pct_exclam = text.count("!") / sentence_len
        pct_question = text.count("?") / sentence_len
        return {"pct_exclam": pct_exclam, "pct_question": pct_question}

    def _ner_counts(self, doc):
        counts = dict.fromkeys(self.ner_cols, 0)
        for ent in doc.ents:
            if ent.label_ in counts:
                counts[ent.label_] += 1
        return counts
    
    def _verb_noun_ratio(self, doc):
        num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
        num_nouns = sum(1 for token in doc if token.pos_ == "NOUN")
        
        ratio = num_verbs / num_nouns if num_nouns > 0 else 0
        return {"verb_noun_ratio": ratio}


class ClassifierWithBinarization(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = CatBoostClassifier(
            loss_function = "MultiClass",
            eval_metric="TotalF1",
            random_seed=42,
            verbose=False,
         )

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        positive = 0.5*probs[:, 2] + probs[:, 3] + 1.5 * probs[:, 4]
        negative = probs[:, 1] + 1.5 * probs[:, 0]

        return (positive - negative > 0).astype(int)

class SentimentFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            blob = TextBlob(str(text))
            feats.append([blob.sentiment.polarity, blob.sentiment.subjectivity])
        return np.array(feats)
    
class HF_Sklearn_Ensemble:
    def __init__(self, hf_model, sklearn_pipeline, hf_ratio=0.5):
        self.hf_model = hf_model
        self.sklearn_pipeline = sklearn_pipeline
        self.hf_ratio = hf_ratio

    def predict_proba(self, X, text_col="text"):
        # --- Make a clean text list ONCE, use it everywhere ---
        texts = X[text_col].astype(str).tolist()
        n = len(texts)

        # HF probabilities
        _, logits = self.hf_model.predict(texts)
        hf_probs = softmax(logits, axis=1)[:, 1]
        if hf_probs.shape[0] != n:
            raise ValueError(f"HF returned {hf_probs.shape[0]} preds but X has {n} rows")

        # Sklearn probabilities (ensure it uses the same X)
        probs = self.sklearn_pipeline.predict_proba(X)
        if probs.shape[0] != n:
            raise ValueError(f"Sklearn returned {probs.shape[0]} preds but X has {n} rows")

        # Your custom mapping
        positive = 0.5 * probs[:, 2] + probs[:, 3] + 1.5 * probs[:, 4]
        negative = probs[:, 1] + 1.5 * probs[:, 0]
        sk_probs = positive / (positive + negative)

        # Ensemble average
        avg_probs = self.hf_ratio * hf_probs + (1 - self.hf_ratio) * sk_probs
        return avg_probs

    def predict(self, X, text_col="text", threshold=0.5):
        avg_probs = self.predict_proba(X, text_col=text_col)
        return (avg_probs >= threshold).astype(int)
