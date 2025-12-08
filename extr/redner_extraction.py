# Extrahiere Rednernamen und alle zugehörigen Redetexte aus dem XML, wie sie in <rede> gespeichert sind
def extract_speeches(root):
    speeches = []
    for rede in root.findall(".//rede"):
        # Rednername aus dem ersten <p klasse="redner"> extrahieren
        redner_p = rede.find("./p[@klasse='redner']")
        if redner_p is not None:
            # Versuche Vorname und Nachname aus verschachtelten Tags zu holen
            name_tag = redner_p.find(".//name")
            if name_tag is not None:
                vorname = name_tag.findtext("vorname", default="").strip()
                nachname = name_tag.findtext("nachname", default="").strip()
                redner_name = f"{vorname} {nachname}".strip()
            else:
                # Fallback: Text direkt aus <p> nehmen
                redner_name = redner_p.text.split(":")[0].strip() if redner_p.text else ""
        else:
            redner_name = ""

        # Alle <p> mit Redeinhalt (ohne klasse="redner" und ohne <kommentar>)
        rede_text = []
        for p in rede.findall("./p"):
            if p.get("klasse") != "redner" and p.text and p.text.strip():
                rede_text.append(p.text.strip())
        full_text = "\n".join(rede_text)
        speeches.append({"name": redner_name, "text": full_text})
    return speeches




def extract_all_speakers(root):
    speakers = []
    for rednerliste in root.findall(".//rednerliste"):
        for redner in rednerliste.findall("redner"):
            name_tag = redner.find("name")
            if name_tag is not None:
                vorname = name_tag.findtext("vorname", default="").strip()
                nachname = name_tag.findtext("nachname", default="").strip()
                fraktion = name_tag.findtext("fraktion", default="").strip()
                rolle_lang = name_tag.findtext("rolle/rolle_lang", default="").strip()
                rolle_kurz = name_tag.findtext("rolle/rolle_kurz", default="").strip()
                speakers.append({
                    #"id": redner.get("id"),
                    "vorname": vorname,
                    "nachname": nachname,
                    "fraktion": fraktion,
                    #"rolle_lang": rolle_lang,
                    "rolle_kurz": rolle_kurz 
                })
                
    return speakers



##############################
##### Sentiment Analysis #####
##############################

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "oliverguhr/german-sentiment-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_average_sentiment(text, chunk_size=512):
    """
    Teilt einen Text in Chunks auf und berechnet das durchschnittliche Sentiment
    """
    sentiments = []
    
    # Text in Chunks aufteilen
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Nur nicht-leere Chunks verarbeiten
            result = sentiment_pipeline(chunk)
            sentiments.append(result)
    
    # Scores extrahieren und Durchschnitt berechnen
    scores = [item[0]['score'] for item in sentiments]
    average_score = sum(scores) / len(scores) if scores else 0
    
    # Label basierend auf Durchschnitt bestimmen
    average_label = sentiments[0][0]['label'] if sentiments else "NEUTRAL"
    
    return {
        'label': average_label,
        'score': average_score,
        'chunks_analyzed': len(sentiments)
        
    }


##############################
#####   Text Embedding   #####  
##############################


from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # klein & schnell; gute Qualität

def embed(texts):
    # texts: str or list[str]
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

def cosine_sim(a, b):
    # numerisch stabil: L2-normalisieren dann dot
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

# # Example
# t1 = "This is an example text."
# t2 = "This is an example text !!!!!!!!!!!!!!!."
# v1, v2 = embed([t1, t2])


# print("Cosine similarity:", cosine_sim(v1, v2))





