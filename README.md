# LLM as an Evaluator

## Overview
This project leverages large language models (LLMs) to evaluate biomedical question-answering datasets. The primary dataset used is [MEDIQA](https://github.com/Andy-jqa/biomedical-qa-datasets?tab=readme-ov-file#mediqa-qa). The goal is to provide a robust evaluation framework for biomedical QA systems.

## File Structure
```plaintext
/llm-as-a-evaluator
├── notebooks
│   ├── data_preprocessing.py
│   └── results.ipynb
├── src
│   └── app.py
├── README.md
├── requirements.txt
```

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/llm-as-a-evaluator.git
    cd llm-as-a-evaluator
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the MEDIQA dataset and place it in the `data/mediqa` directory.
### API Key
To use the LLMs, you need to obtain an API key from Groq. Groq offers free API keys. Set the API key as an environment variable or in a `.env` file.

#### Obtaining Groq API Key
1. Visit the [Groq Console](https://console.groq.com/keys).
2. Sign in or create an account.
3. Navigate to the API Keys section.
4. Generate a new API key and copy it.
5. Set the API key as an environment variable:
    ```bash
    export GROQ_API_KEY=your_groq_api_key_here
    ```
6. Alternatively, create a `.env` file in the project root and add the API key:
    ```plaintext
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Features
- **Data Preprocessing:** Scripts to preprocess the MEDIQA dataset.
- **Model Evaluation:** Tools to evaluate QA models using LLMs.
- **Analysis Notebooks:** Jupyter notebooks for detailed analysis.

## License
This project is licensed under the MIT License - see the [MIT License](https://opensource.org/license/MIT) file for details.

## Additional Information
For more details on the MEDIQA dataset, visit the [official repository](https://github.com/Andy-jqa/biomedical-qa-datasets?tab=readme-ov-file#mediqa-qa).

