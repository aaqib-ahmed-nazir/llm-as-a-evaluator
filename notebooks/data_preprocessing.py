import xml.etree.ElementTree as ET
import pandas as pd

def process_xml():
    """
        parameters: None
        
        returns: None
        
        description:
            - This function reads the MEDIQA2019-TASK3-QA-ValidationSet.xml file and extracts the highest ranked answer for each question.
            - It then saves the results to a CSV file.
    """
    tree = ET.parse('MEDIQA2019-TASK3-QA-ValidationSet.xml')
    root = tree.getroot()
    
    questions = []
    answers = []
    ranks = []
    
    for question in root.findall('.//Question'):
        question_text = question.find('QuestionText').text
        answer_list = question.find('AnswerList')
        
        best_answer = None
        highest_rank = -1
        
        for answer in answer_list.findall('Answer'):
            rank = float(answer.attrib['SystemRank'])
            if rank > highest_rank:
                highest_rank = rank
                best_answer = answer.find('AnswerText').text
        
        if best_answer:
            questions.append(question_text)
            answers.append(best_answer)
            ranks.append(highest_rank)
    
    df = pd.DataFrame({
        'Question': questions,
        'BestAnswer': answers,
        'SystemRank': ranks
    })
    
    df.to_csv('highest_ranked_answers.csv', index=False, encoding='utf-8')
    print(f"Created CSV with {len(df)} question-answer pairs")

if __name__ == "__main__":
    process_xml()