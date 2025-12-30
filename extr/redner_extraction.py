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




### Extraction of comments, interjections, remarks ###

def local_name(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

def tag_matches(tag, kws):
    t = local_name(tag).lower()
    return any(k in t for k in kws)


def extract_comments(root):
    keywords = ['zuruf','zwischenruf','bemerkung','beifall','ruf','rufe','intervention','kommentar','ausruf']

    comments_by_speech = {}
    # find speech elements (tags that contain 'rede' or 'speech' in their local name)
    speech_elems = [e for e in root.iter() if tag_matches(e.tag, ['rede','speech'])]

    for s in speech_elems:
        # determine speaker name from attributes or child elements
        speaker = None
        for attr in ('speaker','name','redner','vorname','nachname'):
            if attr in s.attrib and s.attrib.get(attr):
                speaker = s.attrib.get(attr)
                break
        if not speaker:
            vor = None; nach = None; name = None
            for desc in s.iter():
                ln = local_name(desc.tag).lower()
                if ln == 'vorname' and (desc.text or '').strip():
                    vor = (desc.text or '').strip()
                if ln == 'nachname' and (desc.text or '').strip():
                    nach = (desc.text or '').strip()
                if ln == 'name' and (desc.text or '').strip():
                    name = (desc.text or '').strip()
            if vor and nach:
                speaker = f"{vor} {nach}"
            elif name:
                speaker = name
        if not speaker:
            # fallback: try to match using the beginning of speech text in `reden`
            snippet = ''.join(s.itertext()).strip()[:200]
            speaker = next((n for n,t in reden.items() if snippet and (snippet in t[:200] or t[:200] in snippet)), 'Unknown')

        # collect comment-like descendant elements
        for desc in s.iter():
            if tag_matches(desc.tag, keywords):
                txt = ' '.join(p.strip() for p in desc.itertext() if (p or '').strip())
                if txt:
                    comments_by_speech.setdefault(speaker, []).append(txt)
    return comments_by_speech




##############################
##### Sentiment Analysis #####
##############################

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "oliverguhr/german-sentiment-bert"  # Alternative Modell
#model_name = "ChrisLalk/German-Emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline( "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

def get_average_sentiment(text, chunk_size=512):
    """
    Teilt einen Text in Chunks auf und berechnet das durchschnittliche Sentiment
    """
    sentiments = []
    print(model)
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





