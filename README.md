# Assignment 1: Medical QA

## Project Overview
This project implements both extractive and generative Question
Answering (QA) pipelines over clinical text. The goal is to
evaluate how well pretrained language models can extract
verifiable spans from medical context versus generating
synthesized answers.

## Environment Setup
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


## Task 2: Generative QA Architecture
I implemented a text-to-text generation pipeline requiring manual prompt
engineering to feed both the context and question to the model.

**Model Chosen:**
1.`google/flan-t5-small` as the baseline because the FLAN variants are specifically instruction-tuned. Unlike standard causal
models like GPT-2, T5's encoder-decoder architecture is highly effective at processing a block of context and synthesizing an answer,
making it a robust, lightweight baseline for Generative QA.

## Decoding Experiment: Temperature Scaling
To evaluate how decoding strategies affect generative outputs, I ran `flan-t5-small` across two temperature extremes: 
$T = 0.1$ and $T = 1.5$

## Task 2: Results Table (Decoding Experiment)
*(Note: Showing paired results to demonstrate the effect of temperature scaling on generation)*

| Model | Question | Context Snippet | True Answer | Predicted Answer |
|-------|----------|-----------------|-------------|------------------|
| flan-t5-small (Temp 0.1) | What is the main cause of HIV-1 infection in children? | Functional Genetic Variants in DC-SIGNR... | Mother-to-child transmission (MTCT) is the... | Mother-to-child transmission |
| flan-t5-small (Temp 1.5) | What is the main cause of HIV-1 infection in children? | Functional Genetic Variants in DC-SIGNR... | Mother-to-child transmission (MTCT) is the... | transmission child HIV genetic variant Mother MTCT |
| flan-t5-small (Temp 0.1) | How many children were infected by HIV-1? | Functional Genetic Variants in DC-SIGNR... | more than 400,000 children were infected... | 400,000 |
| flan-t5-small (Temp 1.5) | How many children were infected by HIV-1? | Functional Genetic Variants in DC-SIGNR... | more than 400,000 children were infected... | The children 400 000 infected by HIV 1 virus in |

## Failure Analysis

* **Failure Case 1: Numerical Span Ambiguity (RoBERTa & DistilBERT)**
  * **Question:** How many children were infected by HIV-1?
  * **True Answer:** more than 400,000 children
  * **Predicted Answer:** 97 (RoBERTa) / 12 (DistilBERT)
  * **Hypothesis:** Both models completely failed to extract the correct figure. Extractive models heavily rely on local context around numerical entities. 
  The context likely contains multiple numerical figures (e.g., sample sizes in a study, like "12 patients" or "97 transcripts"). The models falsely anchored onto these 
  local, dense numerical clusters rather than successfully resolving the long-range dependency back to "children infected globally."

* **Failure Case 2: Lexical Degradation (Generative, High Temp)**
  * **Model** `flan-t5-small` (Temp 1.5)
  * **Hypothesis:** At $T=1.5$, the flattened probability distribution forces the model to sample highly unlikely tokens. Instead of
   synthesizing a coherent answer from the context, the model loses the semantic thread of the prompt, resulting in fragmented text or 
   repeated tokens. It fails because the decoding strategy essentially introduces too much "noise" into the generation step, overriding 
   the model's learned language structure.

* **Failure Case 3: Over-Brevity / Truncation (Generative, Low Temp)**
  * **Model:** `flan-t5-small` (Temp 0.1)
  * **Hypothesis:** At $T=0.1$, the model operates almost entirely greedily. While factually safer, generative models at near-zero 
  temperatures often output the absolute minimum number of tokens required to complete the sequence, sometimes resulting in one-word answers 
  that strip away necessary clinical nuance. The mechanism of failure is over-confidence in a single, short token path, completely bypassing 
  alternative, more descriptive phrasing.


