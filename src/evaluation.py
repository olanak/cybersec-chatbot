from typing import List
from rouge_score import rouge_scorer

def faithfulness_score(answer: str, sources: List[str]) -> float:
    """
    Compute a simple faithfulness score: 
    how much the answer overlaps with the sources using ROUGE-L.
    answer: текст ответа модели
    sources: список текстов документов
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    combined_sources = "\n".join(sources)
    
    score = scorer.score(combined_sources, answer)

    return score['rougeL'].fmeasure

