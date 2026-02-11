
import pandas as pd
import re
import html
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

NER_COLS = ["ORG", "GPE", "NORP", "DATE", "CARDINAL", "PRODUCT", "ORDINAL", "LOC", "LAW"]
BASE_COLS = ["keyword", "country"]
LABEL_COL = "PCL_category"

class FeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, nlp, stop_words):
        self.nlp = nlp
        self.ner_cols = NER_COLS
        self.base_cols = BASE_COLS
        self.stop_words = stop_words
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        
        texts = X["text"].astype(str).tolist()
        results = []
        
        for doc in self.nlp.pipe(texts, batch_size=32):
 
            row_features = {
                **self._clean_text(doc),
                **self._punctuation_features(doc),
                **self._ner_counts(doc),
                **self._verb_noun_ratio(doc)
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


def data_preprocess(data_path: str):
    df = pd.read_csv(
        data_path,
        sep="\t",
        skiprows=9,
        engine="python",
        index_col=0,
        header=None,
        names = ["article_id", "keyword", "country", "text", "PCL_category"]
    )
    y_categorical = df[LABEL_COL]
    y_binary = (y_categorical >= 2).astype(int)
    X = df.drop(columns=[LABEL_COL])
    return X, y_binary, y_categorical

def main():
    #parser 
    data_path = "dontpatronizeme_pcl.tsv"
    X, y_binary, y_categorical = data_preprocess(data_path)


    nlp = spacy.load("en_core_web_sm")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
   
    feature_extractor = FeatureExtractor(nlp, stop_words)
    X_transformed = feature_extractor.transform(X)
    print(X_transformed.head())

if __name__ == "__main__":
    main()