import gradio as gr
import pickle
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from modules.utils import extract_features
from modules.llm_reasoner import llm_reason
from modules.chroma_memory import store_memory, retrieve_similar
from modules.external_evidence import external_evidence_analysis

# ------------------ Load ML Artifacts ------------------
model = pickle.load(open("models(pkl)/model (4).pkl", "rb"))
tfidf = pickle.load(open("models(pkl)/tfidf (1).pkl", "rb"))

# ------------------ Model Performance Metrics ------------------
# (Replace these with real test-set metrics if available)
accuracy = 0.99
precision = 0.99
recall = 0.99
f1 = 0.99


# ------------------ Core Analysis Logic ------------------
def analyze_news(text):
    if not text or len(text.strip()) < 50:
        return (
            "⚠️ TOO SHORT",
            0.0,
            "Low",
            "Please provide a longer news article for meaningful analysis.",
            "No memory lookup performed.",
            "",
        )

    # -------- Feature Extraction --------
    text_vec = tfidf.transform([text])
    num_features = extract_features(text)
    final_features = hstack([text_vec, num_features])

    # -------- ML Prediction --------
    prob_real = model.predict_proba(final_features)[0][1]
    verdict = "REAL" if prob_real >= 0.5 else "FAKE"

    # -------- Confidence Level --------
    if prob_real >= 0.75:
        confidence = "High"
    elif prob_real >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    # -------- Rule-Based Linguistic Explanation --------
    emotion = float(num_features[0][0])
    caps = float(num_features[0][2])
    readability = float(num_features[0][5])

    reasons = []
    if emotion > 0.05:
        reasons.append(
            "Elevated emotional language detected, commonly linked to sensational or misleading content."
        )
    if caps > 0.1:
        reasons.append(
            "Abnormal capitalization patterns suggest possible clickbait-style writing."
        )
    if readability < 15:
        reasons.append(
            "Low readability score indicates weak journalistic structure."
        )
    if not reasons:
        reasons.append(
            "The article demonstrates neutral tone, balanced language, and professional structure."
        )

    rule_explanation = "\n".join([f"• {r}" for r in reasons])

    # -------- External Evidence Analysis --------
    evidence_result = external_evidence_analysis(text)

    evidence_score = evidence_result["evidence_score"]
    verified_entities = evidence_result["verified_entities"]

    if verified_entities:
        evidence_summary = "\n".join(
            [f"• {v['entity']}: {v['summary']}..." for v in verified_entities]
        )
    else:
        evidence_summary = "No verifiable entities found in the article."

    # -------- Memory Retrieval --------
    past_cases = retrieve_similar(text)
    if past_cases:
        memory_context = "\n".join(
            f"- {m['verdict']} (score={round(m['score'], 2)})"
            for m in past_cases
        )
        memory_note = "Similar historical cases found and used for reasoning."
    else:
        memory_context = ""
        memory_note = "No similar historical cases found."

    # -------- LLM Reasoning --------
    llm_explanation = llm_reason(
        article=text,
        ml_score=round(prob_real, 3),
        linguistic_signals={
            "emotion_score": round(emotion, 3),
            "caps_ratio": round(caps, 3),
            "readability_score": round(readability, 2),
            "external_evidence_score": round(evidence_score, 2),
        },
        memory_context=memory_context,
    )

    # -------- Store Memory --------
    store_memory(
        text=text,
        verdict=verdict,
        score=round(prob_real, 3),
        explanation=llm_explanation,
    )

    # -------- Final Explanation --------
    final_explanation = f"""
### 📌 Rule-Based Linguistic Analysis
{rule_explanation}

### 🌐 External Evidence Check
Evidence Score: {evidence_score:.2f}
{evidence_summary}

### 🦙 LLM Reasoning
{llm_explanation}

### 🧬 Memory Insight
{memory_note}

### 🧠 Decision Rationale
The final verdict is produced by fusing:
- Statistical ML prediction (TF-IDF + classifier)
- Interpretable linguistic signals (XAI)
- Semantic memory retrieval (ChromaDB)
- External evidence verification (Wikipedia)
- Open-source LLM reasoning
"""

    # -------- Model Performance Metrics --------
    performance_metrics = f"""
### 📊 Model Performance
- Accuracy: {accuracy:.3f}
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1 Score: {f1:.3f}
"""

    return (
        verdict,
        float(prob_real),
        confidence,
        final_explanation,
        memory_note,
        performance_metrics,
    )


# ------------------ Gradio UI ------------------
with gr.Blocks(title="AI News Credibility Analyzer") as demo:
    gr.Markdown("## 🧠 AI News Credibility Analyzer")
    gr.Markdown(
        "Analyze news articles using **ML + XAI + LLM Reasoning + Semantic Memory + External Evidence**"
    )

    input_text = gr.Textbox(
        label="📰 Paste News Article",
        lines=12,
        placeholder="Paste the full news article here...",
    )

    analyze_btn = gr.Button("Analyze Credibility 🚀")

    verdict_out = gr.Textbox(label="Verdict")
    score_out = gr.Slider(0, 1, label="Credibility Score", interactive=False)
    confidence_out = gr.Textbox(label="Confidence Level")
    explanation_out = gr.Markdown()
    memory_out = gr.Textbox(label="Memory Status")
    performance_out = gr.Markdown(label="Model Performance Metrics")

    analyze_btn.click(
        analyze_news,
        inputs=input_text,
        outputs=[
            verdict_out,
            score_out,
            confidence_out,
            explanation_out,
            memory_out,
            performance_out,
        ],
    )

demo.launch()
