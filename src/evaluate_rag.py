import json
import os
from openai import OpenAI   # uses OPENAI_API_KEY env var


SYSTEM_PROMPT = """
You are a strict evaluator of answers in a RAG system.

Given a QUESTION, an ANSWER, and the retrieved CONTEXT,
judge how faithful the ANSWER is to the CONTEXT.

Rules:
- The answer is FAITHFUL only if all factual claims are supported by the CONTEXT.
- If the answer adds new facts not in CONTEXT, or contradicts CONTEXT, lower the score.
- Ignore style and formatting.

Return ONLY a number between 0 and 1:
- 1 = fully faithful, no hallucination.
- 0 = completely unsupported or contradictory.
"""


def judge_faithfulness(question: str, answer: str, context_text: str) -> float:
    client = OpenAI()

    prompt = f"""
QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context_text}

Score the faithfulness of the ANSWER to the CONTEXT (0 to 1). 
Return ONLY the number.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",      # or any other model you use
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content.strip()
    try:
        return float(text)
    except ValueError:
        # if model said something else, default to 0.0
        return 0.0


def main():
    # Load your eval examples from a JSONL file
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "eval_data.jsonl")

    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["question"]
            a = row["answer"]
            ctx = " ".join(row["contexts"])  # merge list into one string
            score = judge_faithfulness(q, a, ctx)
            scores.append(score)
            print(f"{q}\n  → faithfulness: {score:.2f}\n")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nAverage faithfulness over {len(scores)} examples: {avg:.2f}")


if __name__ == "__main__":
    main()
