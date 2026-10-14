# PunBot

A pun-analysis chatbot for the CNIT 58100 group project (Group 8). A sentence goes through three stages:

| Stage | What it does | Tools |
|---|---|---|
| **1. Sense activation** | For each candidate word, compare its BERT vector *in this sentence* with each of its WordNet senses, and find which other word in the sentence triggers each sense. A pun needs two clearly different senses, each triggered by a **different** context word. | BERT (`bert-base-uncased`), WordNet |
| **2. Structural check** | Keep only senses with the word's part of speech (POS integrity) and report the word's grammatical role and how each trigger word connects to it in the dependency parse. | spaCy (`en_core_web_sm`) |
| **3. Coherence and dialog** | Claude receives the sentence plus the Stage 1-2 evidence, checks whether the two senses make a coherent joke, overrules the pipeline when it is wrong (e.g. sound-alike puns), makes the final Pun / Non-pun call, and answers follow-up questions. | Claude Sonnet 5.5 |

Give the chatbot a pun and it will:

1. **Classify** the sentence as Pun or Non-pun.
2. Identify the **pivot** word and whether the pun is homographic (same spelling) or heterographic (sound-alike).
3. Define **both senses** and point to the words that support each one.
4. Run a **coherence check**: does each sense fit its context, or does the combination fall apart?
5. Give a **verdict** (Works, Weak or Fails).
6. Say whether it **agrees with Stages 1-2**, and why not when it doesn't.

The chat window shows a collapsible "Stages 1-2" panel with the pipeline's pivot, senses, trigger words and scores. Questions and follow-ups ("Why is it funny?") go straight to Claude with the conversation history.

## How it works

```
Browser ──POST /chat──▶ FastAPI (backend/main.py) ──▶ PunBot (backend/punbot.py)
                                                        │
                    candidate sentence? ── yes ──▶ Stages 1-2 (backend/pipeline/)
                                                        │   spaCy parse → WordNet senses (POS-filtered)
                                                        │   → BERT activation + trigger words → evidence
                                                        ▼
                                                  Stage 3: Claude (backend/llm_adapter.py)
                                                        │   system prompt: backend/prompts.py
                                                        ▼
                                                  answer + evidence ──▶ chat window, SQLite log
```

### Stage 1 scoring, in detail

For each content word (nouns, verbs, adjectives, adverbs; very general verbs like *be*, *go*, *get* are skipped; particle verbs like *branched out* also get the multiword's senses):

- **Activation** of a sense = cosine between the word's BERT vector in the sentence and the headword's vector inside its WordNet gloss ("interest: a fixed charge for borrowing money...").
- **Support** of a sense = the highest cosine between any other content word in the sentence and the gloss as a whole. That word is the sense's **trigger**.
- **Sense A** = the sense with the best average of activation and support. **Sense B** = the best sense that is clearly different from A (different WordNet category, or path similarity below 0.2) and has a *different* trigger word.
- **Double-sense score** = the lower of the two senses' scores. The Stage 1-2 call is **Pun** when the best word's double-sense score reaches `PUN_THRESHOLD`.

Vectors are the mean of BERT's last four layers. **Anisotropy correction:** raw BERT cosines are high for almost any pair of words (all vectors share a common direction), so a fixed cutoff like 0.70 can't separate senses. Scores subtract a background mean vector (from 30 everyday sentences in `backend/pipeline/encoder.py`) before the cosine. Both raw and corrected values are reported.

Known limits by design: WordNet only knows words as spelled, so **heterographic puns** (coop / coupe, sole / soul, two / too) can't be found in Stages 1-2. Claude can still catch them in Stage 3, which makes them good examples of what each stage contributes.

| File | Purpose |
|---|---|
| `backend/pipeline/stages.py` | Stages 1-2: candidates, POS filter, sense scoring, dependency relations, evidence text |
| `backend/pipeline/encoder.py` | BERT vectors and the anisotropy correction |
| `backend/punbot.py` | Runs the stages in order; decides when a message is a sentence to analyze |
| `backend/prompts.py` | Claude's instructions (pun definition, analysis format, Stage 3 role). **Edit this to change the bot's behavior.** |
| `backend/llm_adapter.py` | The only file that calls the Claude API |
| `backend/analysis.py` | Reads the classification, pivot and Stage 1-2 check back out of Claude's answer |
| `backend/main.py` | `/chat` and `/health` endpoints; serves the web page |
| `evaluation/run_eval.py` | Runs the test set through each system and fills in the workbook |
| `frontend/` | Chat page (plain HTML/CSS/JS) |
| `tests/` | Pytest suite; Claude and BERT are replaced by stand-ins, so tests need no API key or model download |

## Setup

Requires Python 3.10+ and an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).

```bash
git clone https://github.com/alizarogers/cnit_581000_project.git
cd cnit_581000_project

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # includes PyTorch, so this takes a few minutes

cp .env.example .env              # then open .env and paste your key after ANTHROPIC_API_KEY=
```

The first run downloads BERT (about 440 MB, from Hugging Face) and WordNet (about 10 MB); after that they load from your local cache.

## Run

```bash
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. The footer shows whether Stages 1-2 loaded. If spaCy, BERT or WordNet can't load, the app falls back to Claude alone and says why. Set `PIPELINE_MODE=llm_only` in `.env` to turn the pipeline off on purpose.

## Test

```bash
pytest
```

## Evaluation

The test set is `evaluation/Group8_Pun_Test_Set.xlsx` (15 puns, 15 non-puns, 12 dialog questions).

```bash
python -m evaluation.run_eval --pipeline-only               # Stages 1-2 only: free, no API calls
python -m evaluation.run_eval --only detection --limit 3    # quick check that your key works
python -m evaluation.run_eval                              # full system + dialog questions
python -m evaluation.run_eval --compare                    # also Claude alone, for comparison
```

The original workbook is never changed. A filled copy is saved to `evaluation/results/` (gitignored) with:

- **Test Set tab**: column J ("Pipeline says") gets the full system's Pun / Non-pun call, so column K and the Summary tab score themselves. Use `--column M` to write it somewhere else.
- **Dialog Questions tab**: a new "Chatbot answer" column. Each question is asked as a follow-up after the pun has been analyzed. Raters fill in the yellow score cells.
- **Chatbot Output tab**: for every sentence, the Stage 1-2 call, pivot, double-sense score, Sense A and B with trigger words, the pivot's grammatical role, Claude's call and pivot, whether Claude agreed with Stages 1-2, and the full analyses.
- **System Comparison tab**: TP / FP / FN / TN, precision, recall, F1, accuracy, and homographic vs heterographic recall for Stages 1-2 alone, the full system, and (with `--compare`) Claude alone.

A `.json` file with every raw output is saved alongside. Sentences are sent exactly as written, with no labels or hints.

**About the threshold:** `PUN_THRESHOLD` (default 0.30) was chosen by looking at Stage 1-2 scores on this same test set, so the Stages 1-2 numbers are optimistic. Say so in the write-up, or set the threshold before looking at a new test set.

### Chat logs

Every exchange in the web chat is saved to `logs/chat_log.db`, including the Stage 1-2 call and pivot. Export it to CSV:

```bash
sqlite3 -header -csv logs/chat_log.db "select * from chat_log" > chat_log.csv
```

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Never commit this. `.env` is gitignored. |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Any Claude model ID. |
| `PIPELINE_MODE` | `full` | `full` = Stages 1-3; `llm_only` = Claude alone. |
| `BERT_MODEL` | `bert-base-uncased` | Any Hugging Face BERT-style model name, or a local folder. |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline used for Stage 2. |
| `PUN_THRESHOLD` | `0.30` | Double-sense score needed for a Stage 1-2 Pun call. |
| `MAX_SENSES` | `8` | Most frequent WordNet senses scored per word. |
| `MAX_TOKENS` | `1024` | Max length of each answer. |
| `MAX_HISTORY_TURNS` | `10` | Prior question/answer pairs sent for context. |
| `LOG_DB_PATH` | `./logs/chat_log.db` | Where the conversation log is written. |
