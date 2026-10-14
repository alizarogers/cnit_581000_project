# PunBot

A chatbot for the CNIT 58100 group project that answers questions about puns using the Claude API (Claude Sonnet 5.5).

Give it a pun and it will:

1. **Classify** the sentence as Pun or Non-pun.
2. Identify the **pivot** word or phrase and say whether the pun is homographic (same spelling) or heterographic (sound-alike).
3. Define **both senses** and point to the words in the sentence that support each one.
4. Run a **coherence check**: does each sense fit its context, or does the combination fall apart?
5. Give a **verdict** (Works, Weak or Fails) with a reason.

You can then ask open-ended follow-ups ("Why is it funny?", "What knowledge does it need?", "Rewrite it so it lands better"). The conversation history is sent with every question, so the bot knows which pun "it" refers to.

## How it works

```
Browser (frontend/)  ──POST /chat──▶  FastAPI (backend/main.py)
                                        │  system prompt: backend/prompts.py
                                        ▼
                                  Claude Messages API (backend/llm_adapter.py)
                                        │
                                        ▼
                                  SQLite log (logs/chat_log.db)
```

| File | Purpose |
|---|---|
| `backend/prompts.py` | The system prompt: our pun definition and the analysis steps. **Edit this to change the bot's behavior.** |
| `backend/llm_adapter.py` | Calls Claude. Only file that talks to the API. |
| `backend/main.py` | `/chat` and `/health` endpoints; serves the web page. |
| `backend/logger.py` | Logs every question and answer to SQLite for evaluation. |
| `backend/analysis.py` | Reads the Pun / Non-pun classification and pivot word back out of an answer. |
| `evaluation/run_eval.py` | Runs the test set workbook through the bot and fills in the results. |
| `frontend/` | Chat page (plain HTML/CSS/JS). |
| `tests/` | Pytest suite; Claude is mocked, so tests need no API key. |

## Setup

Requires Python 3.10+ and an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).

```bash
git clone https://github.com/alizarogers/cnit_581000_project.git
cd cnit_581000_project

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then open .env and paste your key after ANTHROPIC_API_KEY=
```

## Run

```bash
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 and paste a pun. API docs are at http://127.0.0.1:8000/docs.

## Test

```bash
pytest
```

## Evaluation

The test set lives in `evaluation/Group8_Pun_Test_Set.xlsx` (15 puns, 15 non-puns, 12 dialog questions). Run it through the bot with:

```bash
python -m evaluation.run_eval                              # full run, about 40 API calls
python -m evaluation.run_eval --only detection --limit 3   # quick check that your key works
```

The original workbook is never changed. A filled copy is saved to `evaluation/results/` (gitignored) with:

- **Test Set tab**: column J ("Pipeline says") filled with the bot's Pun / Non-pun call, so column K and the Summary tab score themselves (precision, recall, F1, homographic vs. heterographic recall). Use `--column M` to write the predictions somewhere else instead.
- **Dialog Questions tab**: a new "Chatbot answer" column. Each question is asked as a follow-up after the bot has analyzed that pun, the same way a user would chat. Raters then fill in the yellow score cells.
- **Chatbot Output tab**: the full analysis, parsed pivot word and classification for every sentence, for digging into misses.

A `.json` file with the raw responses is saved alongside. Sentences are sent exactly as written (no labels or hints), at temperature 0 so reruns are comparable.

How detection works: the system prompt asks Claude to start every analysis with `**Classification**: Pun` or `**Classification**: Non-pun`, and `backend/analysis.py` reads that line back. It counts as a pun only if two distinct senses are each supported by context, so ambiguous-but-not-funny sentences ("I went to the bank.") should come back Non-pun.

### Chat logs

Every exchange in the web chat is saved to `logs/chat_log.db`. Export it to CSV:

```bash
sqlite3 -header -csv logs/chat_log.db "select * from chat_log" > chat_log.csv
```

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Never commit this. `.env` is gitignored. |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Any Claude model ID. |
| `MAX_TOKENS` | `1024` | Max length of each answer. |
| `MAX_HISTORY_TURNS` | `10` | How many prior question/answer pairs are sent for context. |
| `LOG_DB_PATH` | `./logs/chat_log.db` | Where the conversation log is written. |
