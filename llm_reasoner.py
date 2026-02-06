from transformers import pipeline

# ------------------ Load LLM ------------------
# Lightweight & HF Spaces friendly
llm = pipeline(
    task="text-generation",
    model="Qwen/Qwen1.5-1.8B",
    max_new_tokens=200,
    temperature=0.25,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

# ------------------ LLM Reasoning ------------------
def llm_reason(article, ml_score, linguistic_signals, memory_context=""):
    """
    Generates a human-readable credibility explanation using an open-source LLM
    """

    prompt = f"""
You are an AI system specialized in news credibility analysis.

Analyze the article ONLY based on writing style, tone, and structure.
Do NOT verify real-world facts.

Inputs:
- ML Credibility Score (0 to 1): {ml_score}
- Linguistic Signals: {linguistic_signals}
- Past Similar Cases (if any): {memory_context}

Article:
\"\"\"{article}\"\"\"

Instructions:
- Be concise and logical
- Avoid hallucinations
- Explain WHY the article seems real or fake
- End with a clear verdict

Format:
Explanation:
- ...
- ...
Verdict: REAL or FAKE
"""

    output = llm(prompt)[0]["generated_text"]

    # Remove prompt echo if present
    cleaned_output = output.replace(prompt, "").strip()

    return cleaned_output
