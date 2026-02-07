import wikipediaapi
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Wikipedia API (User-Agent is mandatory)
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AI-News-Credibility-Analyzer/1.0 (academic project)"
)

def extract_entities(text):
    """
    Extract named entities relevant for verification
    """
    doc = nlp(text)
    return list(
        set(ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"])
    )

def external_evidence_analysis(text):
    """
    Verifies extracted entities using Wikipedia.
    Returns structured evidence data (NOT UI strings).
    """
    entities = extract_entities(text)
    verified_entities = []

    for entity in entities[:3]:  # limit to avoid API overload
        page = wiki.page(entity)
        if page.exists():
            verified_entities.append({
                "entity": entity,
                "summary": page.summary[:300]
            })

    evidence_score = round(len(verified_entities) / max(len(entities), 1), 2)

    return {
        "evidence_score": evidence_score,
        "verified_entities": verified_entities,
        "sources_found": len(verified_entities)
    }
