from src.rag_chain import rewrite_query, answer_question


original_question = "What is Legal, regulatory, and contractual requirements?"


expanded_question = rewrite_query(original_question)
print("Original question:", original_question)
print("Expanded question:", expanded_question)


response = answer_question(original_question, use_rag=True, use_rewrite=True)
print("\nAnswer:", response["answer"])
print("Sources:", response["sources"])

