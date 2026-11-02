# Transcript Intelligence

Load call transcripts, classify them with a local LLM, and review the results in a notebook and dashboard.

Each meeting folder under `data/raw/` contains `transcript.json` and `meeting-info.json`. `src/pipeline.py` loads those files, sends each transcript to a local Ollama model (`llama3.1`), and writes:

- `data/results.csv` — categorized meetings (call type, topic, sentiment, escalation risk, products, confidence)
- `data/failed.csv` — meetings the model call or validator rejected
- `data/cache/` — one JSON file per meeting so reruns skip calls that already succeeded. Pass `--force` to ignore the cache and call the model again.

## Layout

```
transcript-intelligence/
├── data/
│   ├── raw/            # One folder per meeting
│   ├── cache/          # Cached LLM responses
│   ├── results.csv
│   └── failed.csv
├── src/
│   ├── load_data.py    # Read transcript + meeting info
│   ├── analyze.py      # Call Ollama and validate the JSON
│   └── pipeline.py     # Load, analyze, write CSVs
├── notebooks/
│   └── analysis.ipynb  # Categories, sentiment, and extra insights
├── app/
│   └── dashboard.py    # Streamlit view of results.csv
├── tests/
│   └── test_core.py    # Loader, validator, cache, and call-type review
├── requirements.txt
└── README.md
```

## Setup

Ollama needs to be running locally with `llama3.1` pulled:

```bash
brew install ollama
brew services start ollama
ollama pull llama3.1
```

Then install the Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/pipeline.py
python src/pipeline.py --force
jupyter notebook notebooks/analysis.ipynb
streamlit run app/dashboard.py
```

The pipeline talks to `http://localhost:11434` and waits up to 120 seconds per meeting. A full pass over 100 transcripts takes a while. Later runs reuse `data/cache/` unless you pass `--force`, which calls the model again and overwrites each meeting's cache file.

## Tests

From the project root, with the virtualenv active:

```bash
python -m unittest discover -s tests
```

`tests/test_core.py` has six checks and does not call Ollama:

- A single email domain is an internal meeting, two domains is external, and a support-case title is support. A churn signal in `summary.json` is picked up.
- A folder missing `transcript.json` is counted as failed and the complete folder is still loaded.
- A full model response passes validation.
- A missing field, a sentiment score outside -1 to 1, or a call type other than support, external, or internal is rejected.
- A cached JSON file is returned as-is, and the model is not called.
- The dashboard review moves a same-domain meeting to internal, a support-case title to support, and an account call with a customer to external. Product aliases such as "Detect" and "Aegis Detect" collapse to one name.
