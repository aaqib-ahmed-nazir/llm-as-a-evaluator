from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import pandas as pd
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define evaluation criteria
EVALUATION_CRITERIA = {
    "Correctness": "Is the AI answer factually accurate and free of errors?",
    "Relevance": "Does the AI answer directly address the question?",
    "Completeness": "Does the AI answer cover all necessary aspects of the question?",
    "Clarity": "Is the AI answer clearly and coherently presented?",
    "Style and Tone": "Is the AI answer appropriate in style and tone for a medical context?",
    "Harmfulness": "Does the AI answer avoid any harmful or unethical content?"
}

def setup_llm_answerer():
    return ChatGroq(
        model_name="llama-3.1-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )
    
def setup_llm_evaluator():
    return ChatGroq(
        model_name="llama-3.1-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )

def load_data():
    # Load the CSV data
    return pd.read_csv('highest_ranked_answers.csv')

def generate_ai_answer(llm, question):
    # Generate AI answer to the question
    prompt = f"""You are a medical expert. Please provide a professional and accurate answer to the following question:

Question: {question}

Provide a clear, concise, and comprehensive response."""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content.strip()

def evaluate_answers(llm, question, human_answer, ai_answer):
    # Create evaluation prompt
    criteria_list = "\n".join([f"- **{k}**: {v}" for k, v in EVALUATION_CRITERIA.items()])
    evaluation_prompt = f"""As an expert evaluator, assess the AI-generated answer compared to the human answer for the given question based on the following criteria. For each criterion, provide a score between 1 (worst) and 5 (best), along with a brief justification.

**Question:**
{question}

**Human Answer:**
{human_answer}

**AI Answer:**
{ai_answer}

**Evaluation Criteria:**
{criteria_list}

**Your Evaluation:**"""

    messages = [HumanMessage(content=evaluation_prompt)]
    evaluation_response = llm.invoke(messages)
    evaluation_text = evaluation_response.content.strip()
    return parse_evaluation(evaluation_text)

def parse_evaluation(evaluation_text):
    # Parse the evaluation to extract scores
    scores = {}
    total_score = 0
    for criterion in EVALUATION_CRITERIA.keys():
        pattern = rf"{criterion}.*?Score.*?(\d)"
        match = re.search(pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
        if match:
            score = int(match.group(1))
            scores[criterion] = score
            total_score += score
        else:
            scores[criterion] = None  # Score not found

    # Normalize total score to a 1-10 scale
    max_total = 5 * len(EVALUATION_CRITERIA)
    if total_score:
        normalized_score = (total_score / max_total) * 10
        normalized_score = round(normalized_score, 2)
    else:
        normalized_score = None
    return {
        'scores': scores,
        'total_score': normalized_score,
        'evaluation_text': evaluation_text
    }

def main():
    # Initialize LLM
    setup_llm_answerer = setup_llm_answerer()
    setup_llm_evaluator = setup_llm_evaluator()

    # Load data
    df = load_data()
    results = []

    # Evaluate each question-answer pair
    for index, row in df.iterrows():
        question = row['Question']
        human_answer = row['BestAnswer']
        print(f"Processing Question {index + 1}/{len(df)}")

        # Generate AI answer
        ai_answer = generate_ai_answer(setup_llm_answerer, question)

        # Evaluate AI answer
        evaluation = evaluate_answers(setup_llm_evaluator, question, human_answer, ai_answer)
        scores = evaluation['scores']
        total_score = evaluation['total_score']

        # Compile result
        result = {
            'Question': question,
            'Human_Response': human_answer,
            'AI_Response': ai_answer,
            'AI_Ranking': total_score
        }
        result.update(scores)
        results.append(result)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('qa_evaluation_results.csv', index=False)
    print("Evaluation complete! Results saved to qa_evaluation_results.csv")

if __name__ == "__main__":
    main()