# Assignment 1: Medical QA

## Project Overview
This project implements both extractive and generative Question
Answering (QA) pipelines over clinical text. The goal is to
evaluate how well pretrained language models can extract
verifiable spans from medical context versus generating
synthesized answers.

** Setup Instructions **
1. Clone the repository.
2. Create and activate the virtual environment:
   `uv venv`
   `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
3. Install the exact pinned dependencies:
   `uv pip install -r requirements.txt`
4. Run the full evaluation pipeline:
   `python experiments/run_baseline.py`

## Task 1: Extractive QA Architecture 
I utilized the high-level Hugging Face pipeline abstraction while
using a stable version of the pipeline abstraction, since this 
ensures a cleaner evaluation logic.

**Models Chosen:** 
1. `deepset/roberta-base-squad2` as the primary baseline, since 
RoBERTa is heavily optimized for extractive tasks and the 
deepset variant is specifically fine-tuned for high-stakes
context-QA.
2. `distilbert-base-cased-distilled-squad` as a comparative 
baseline to test if a much smaller, faster distilled model
could maintain clinical span extraction accuracy compared
to RoBERTa.

## Result Table
(Note: Showing a subnet of results from 
`results/task1_extractive_results.csv`)

| Model | Question | Context Snippet | True Answer | Predicted Answer |
|-------|----------|-----------------|-------------|------------------|
| RoBERTa | How many children were infected by HIV-1? | Functional Genetic Variants in... | more than 400,000 children... | 97 |
| DistilBERT | How many children were infected by HIV-1? | Functional Genetic Variants in... | more than 400,000 children... | 12 |
| RoBERTa | What inhibits S-palmitoylation? | Role of S-Palmitoylation on IFITM5... | 2-bromopalmitic acid (2BP) | 2BP |

## Failure Analysis

* **Failure Case 1: Numerical Span Ambiguity (RoBERTa & DistilBERT)**
  * **Question:** How many children were infected by HIV-1?
  * **True Answer:** more than 400,000 children
  * **Predicted Answer:** 97 (RoBERTa) / 12 (DistilBERT)
  * **Hypothesis:** Both models completely failed to extract the correct figure. Extractive models heavily rely on local context around numerical entities. 
  The context likely contains multiple numerical figures (e.g., sample sizes in a study, like "12 patients" or "97 transcripts"). The models falsely anchored onto these 
  local, dense numerical clusters rather than successfully resolving the long-range dependency back to "children infected globally."



