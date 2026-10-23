# CASL Analysis Tool

Speech-language analysis for SLPs, organized around the domains of the **CASL-2** (Comprehensive
Assessment of Spoken Language): lexical/semantic, syntactic, supralinguistic and pragmatic. Upload
audio, video or a CHAT transcript. The app transcribes it, computes language-sample metrics, and
asks Claude for a structured, evidence-quoted analysis that a clinician then reviews. Deployed as a
Hugging Face Space (`simple_casl_app.py`).

## What it does

- **Input:** audio or video (the audio is extracted from video), speaker diarization, or CHAT-format
  transcripts (`*PAR`/`*INV`, `&-um`, `[/]`, `[//]`, `[*]`).
- **Metrics:** disfluencies, repetitions and revisions, word-retrieval and grammatical errors from the
  transcript (`calculate_slp_metrics`).
- **LLM analysis:** a sectioned prompt over the transcript, age, gender and SLP notes. Long answers
  are continued across calls and reassembled section by section. Every finding must quote the
  transcript.
- **Targeted questions:** ask a custom question about the same sample.
- **Reports:** PDF export and patient records in the `moderate` and `full` variants.

> **Not a standardized score.** The CASL-2 is a normed, published test. This tool does not
> administer it, does not contain its items or norms, and any score-like numbers it shows are LLM
> estimates. It supports a clinician's analysis and does not replace the test.

## App variants

`simple_casl_app.py` is the deployed app. The others are earlier or experimental builds:
`moderate_casl_app.py` (transcription + PDF), `full_casl_app.py` (visualizations, records, AWS
Bedrock backend), `experimental_casl_app.py`, `annotated_casl_app.py`, and `enhanced_casl_app.py`.
`notebooks/casl_analysis_colab.ipynb` is the original Colab version, and `transcription_demo.py`
demonstrates the transcription step on its own.

## Voice bot (`voicebot/`)

A conversational version. An AI speech pathologist talks with the student and tracks the CASL-2
domains live. There are three implementations: LiveKit agents with OpenAI's realtime voice API, direct
OpenAI API calls, and an adaptive Hugging Face Spaces build. Sessions can be saved and the SLP can
take notes. See `voicebot/README.md`.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python simple_casl_app.py
```

No patient data is included. Research and education tool, not a medical device.
