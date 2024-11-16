from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import pandas as pd
import time
import os
from dotenv import load_dotenv
import re

"""
    Load environment variables from a .env file
    - GROQ_API_KEY: API key for the GROQ model
"""
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

"""
    Defining evaluation criteria for the AI-generated answers
"""
EVALUATION_CRITERIA = {
    "Correctness": "Is the AI answer factually accurate, medically sound, and free from errors?",
    "Relevance": "Does the AI answer directly and specifically address the question posed, staying on-topic?",
    "Completeness": "Does the AI answer thoroughly cover all relevant aspects, including treatment options, risks, or other necessary details?",
    "Clarity and Coherence": "Is the AI answer clearly structured, logically organized, and easy to understand for a medical professional audience?",
    "Professional Tone and Style": "Is the AI answer professional, maintaining a formal tone and suitable clinical terminology appropriate for an expert medical forum?",
    "Ethical and Harm-Free": "Does the AI answer avoid any harmful, misleading, or ethically inappropriate content, prioritizing patient safety and accuracy?"
}


def setup_llm_answerer():
    """
        parameters: None
        
        returns: ChatGroq object
        
        description: This function sets up the GROQ model for generating AI answers to medical questions.
    """
    return ChatGroq(
        model_name="llama-3.1-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )
        
def setup_llm_evaluator():
    """
        parameters: None
        
        returns: ChatGroq object
        
        description: This function sets up the GROQ model for evaluating AI-generated answers.
    """
    return ChatGroq(
        model_name="llama-3.1-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )

def load_data():
    """
        parameters: None
        
        returns: pd.DataFrame
        
        description: This function loads the data containing medical questions and human answers for evaluation.
    """
    return pd.read_csv('highest_ranked_answers.csv')

def generate_ai_answer(llm, question):
    """
        parameters:
            - llm: ChatGroq object
            - question: str
            
        returns:
            - response: str
            
        description:
            - This function generates an AI answer to a given medical question using the GROQ model.
            - It formats the question as a prompt for the AI model to generate a detailed and informative answer.
    """
    prompt = f"""As a medical expert responding on a professional medical forum, provide a well-structured, detailed answer to the following question, aiming for a response length of 100 to 700 words. Your answer should reflect expert-level knowledge, using precise clinical terminology and evidence-based information. Focus solely on relevant medical details, addressing potential treatments, options, and considerations. Avoid casual language, greetings, or unnecessary context. Structure your response as if offering peer-level advice in a professional setting.:

    Question: {question}

    Your answer should be informative but succinct, similar to the provided example."""

    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return response.content.strip()

def evaluate_answers(llm, question, human_answer, ai_answer):
    """
        parameters:
            - llm: ChatGroq object
            - question: str
            - human_answer: str
            
        returns:
            - score: float
            
        description:
            - This function evaluates the AI-generated answer against the human answer for a given question based on predefined evaluation criteria.
            - It prompts the evaluator to provide a numerical score reflecting the quality of the AI answer compared to the human answer.
    """
    
    criteria_list = "\n".join([f"- **{k}**: {v}" for k, v in EVALUATION_CRITERIA.items()])
    evaluation_prompt = f"""Evaluate the AI-generated answer against the human answer for the given question.

    Question:
    {question}

    Human Answer:
    {human_answer}

    AI Answer:
    {ai_answer}

    **Evaluation Criteria:**
    {criteria_list}

    On a scale from 1 to 10, where 1 is the worst and 10 is the best, provide a single numerical score reflecting the quality of the AI answer compared to the human answer.

    Only provide the numerical score."""

    messages = [HumanMessage(content=evaluation_prompt)]
    
    try:
        evaluation_response = llm.invoke(messages)
    except Exception as e:
        print(f"An error occurred: {e}")
        
    evaluation_text = evaluation_response.content.strip()

    # Extract the numerical score
    match = re.search(r'\b(\d{1,2}(?:\.\d+)?)\b', evaluation_text)
    if match:
        score = float(match.group(1))
    else:
        score = None 
    return score

def main():
    """
        parameters: None
        
        returns: None
        
        description: 
            - This function orchestrates the evaluation of AI-generated answers to medical questions.
            - It loads the data, generates AI answers, evaluates them, and saves the results to a CSV file.
    """
    llm_answerer = setup_llm_answerer()
    llm_evaluator = setup_llm_evaluator()

    # Loading data
    df = load_data()
    results = []

    # Evaluating
    for index, row in df.iterrows():
        question = row['Question']
        human_answer = row['BestAnswer']
        print(f"Processing Question {index + 1}/{len(df)}")

        ai_answer = generate_ai_answer(llm_answerer, question)

        score = evaluate_answers(llm_evaluator, question, human_answer, ai_answer)

        result = {
            'Question': question,
            'Human_Response': human_answer,
            'AI_Response': ai_answer,
            'AI_Ranking': score
        }
        results.append(result)
        time.sleep(10) # Sleep to avoid rate limit :( 

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('qa_evaluation_results.csv', index=False)
    print("Evaluation complete! Results saved to qa_evaluation_results.csv")

if __name__ == "__main__":
    main()