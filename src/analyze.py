import json
import ollama
import hashlib
import os
import requests


REQUIRED_FIELDS = ["call_type", "primary_topic", "secondary_topics", "category", "sentiment_score", "sentiment_label", "escalation_risk", "summary", "evidence","is_business_call","products_mentioned","confidence_score"]
VALID_CALL_TYPES = ["support", "external", "internal"]


model = "llama3.1"


# ── cache setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


PROMPT = """
You are analyzing a business call transcript.

Based on the transcript below, return a JSON object with exactly these fields:

{{
  "call_type": "support" | "external" | "internal",
  "primary_topic": "The main subject discussed e.g. billing dispute, renewal discussion, deployment issue",
  "secondary_topics": ["topic1", "topic2", "topic3"], // Other important themes — clean, deduplicated, standardized
  "category": "one of: Billing, Technical Support, Account Management, Incident Response, Product Feedback, Internal Planning, Onboarding, Other",
  "sentiment_score": a number from -1.0 (very negative) to 1.0 (very positive),
  "sentiment_label": "positive" | "neutral" | "negative",
  "escalation_risk": true | false,
  "is_business_call": true | false,
  "products_mentioned": ["product1", "product2"],
  "confidence_score": a number from 0.0 to 1.0 indicating how confident you are in your analysis of this transcript
  "summary": "2 sentence max summary of what happened",
  "evidence": "Specific excerpts from the transcript supporting evidence for primary_topic, secondary_topics, and sentiment"
}}

Rules:
- call_type support   = customer calling in with a problem or question
- call_type external  = someone from the host company talking TO a customer (demos, renewals, onboarding, account reviews)look at email domains — different domains = external
- call_type internal  = team members talking to each other, no customer present (ALL participants share the same email domain)
- escalation_risk true = customer was frustrated, issue unresolved, or involves billing/legal dispute
- is_business_call false = personal calls, test calls, accidental recordings, calls under 2 minutes, calls with no clear business topic
- products_mentioned = Aegis products referenced in the transcript e.g. ["Aegis Identity", "Aegis Detect", "Aegis Protect", "Comply v2"]. Return empty list if no specific products mentioned

Category rules:
- Billing = invoice questions, payment disputes, seat counts
- Technical Support = customer-reported bugs, configuration help
- Account Management = renewals, QBRs, relationship calls
- Internal Planning = team syncs, roadmap, competitive reviews, post-mortems
- Incident Response = active outage calls, escalation bridges, RCAs
- Onboarding = new customer setup, product walkthroughs
- Product Feedback = feature requests, customer input sessions

Secondary Topics rules :
- Standardize capitalization 
- Merge near-duplicates into one clean label (e.g. "Comply v2 -" and "Comply v2 on-demand reporting" → "Comply v2")
- Return a clean deduplicated list of topic labels
- Return only a JSON array of strings. No explanation. Example: ["Detect outage", "Comply v2", "HIPAA compliance"]

Confidence score rules:
- 1.0 = transcript is clear, complete, easy to categorize
- 0.5 = some ambiguity in call type or sentiment
- 0.0 = transcript is too short, unclear, or missing context

Return only valid JSON. No explanation. No markdown code fences. Do not invent information that is not supported by the transcript.

TRANSCRIPT:
{transcript_text}
"""




def _cache_path(folder):
    return os.path.join(CACHE_DIR, f"{folder}.json")

def _load_cache(folder):
    path = _cache_path(folder)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def _save_cache(folder, result):
    with open(_cache_path(folder), "w") as f:
        json.dump(result, f, indent=2)





def validate_llm_result(result):
    """
    Check the model response is usable.
    Returns (is_valid, reason)
    """
    for field in REQUIRED_FIELDS:
        if field not in result:
            return False, f"Missing required field: {field}"

    if result.get("call_type") not in VALID_CALL_TYPES:
        return False, f"Invalid call_type: {result.get('call_type')}"

    if not isinstance(result.get("escalation_risk"), bool):
        return False, f"escalation_risk must be true/false"

    if result.get("sentiment_score") < -1.0 or result.get("sentiment_score") > 1.0:
        return False, f"sentiment_score must be between -1.0 and 1.0"

    if result.get("sentiment_label") not in ["positive", "neutral", "negative"]:
        return False, f"sentiment_label must be positive/neutral/negative"

    if not isinstance(result.get("is_business_call"), bool):
        return False, "is_business_call must be true/false"
        
    confidence = result.get("confidence_score")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False, f"Invalid confidence_score: {confidence}"
    else:
        return False, "confidence_score is required"

    if not isinstance(result.get("products_mentioned", []), list):
        return False, "products_mentioned must be a list"
    
    return True, "ok"

def call_llm(meeting, force=False):
    """
    Send one meeting transcript to a local Ollama model.
    Returns the parsed JSON result or None if it fails.
    """
    folder = meeting["folder"]
    if not force:
        cached = _load_cache(folder)
        if cached is not None:
            print(f"  [cache] {folder}")
            return cached

    # cache miss (or forced) → call LLM
    transcript = meeting["transcript"]
    prompt = PROMPT.format(transcript_text=transcript)
    print("running for meeting", meeting["meeting_id"])
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )

    text = response.json()["response"].strip()
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```").strip()
    result = json.loads(text)
    _save_cache(folder, result)
    return result

def analyze_call(meetings, force=False):
    results = []
    failed = []
    for meeting in meetings:
        try:
            response = call_llm(meeting, force=force)
            if not response:
                print(f"No response from Ollama for meeting {meeting['meeting_id']}")
                continue
            is_valid, reason = validate_llm_result(response)
            if not is_valid:
                print(f"Invalid LLM result: {reason}")
                failed.append(meeting)
                continue
        except Exception as e:
            print(f"Error calling llm: {e}")
            failed.append(meeting)
            continue

        # Merge meeting metadata with LLM result
        combined = {
            # Metadata from load step
            "meeting_id":     meeting["meeting_id"],
            "title":          meeting["title"],
            "host":           meeting["host"],
            "start_time":     meeting["start_time"],
            "end_time":       meeting["end_time"],
            "duration":       meeting["duration"],
            "attendees":      meeting["attendees"],
            "date":           meeting["start_time"][:10],
            "duration_min":   meeting["duration"],
            "sentence_count": meeting["sentence_count"],

            # From LLM
            "call_type":       response["call_type"],
            "primary_topic":   response["primary_topic"],
            "secondary_topics": response["secondary_topics"],
            "category":        response["category"],
            "sentiment_score": response["sentiment_score"],
            "sentiment_label": response["sentiment_label"],
            "escalation_risk": response["escalation_risk"],
            "llm_summary":     response["summary"],
            "evidence":        response["evidence"],
            "is_business_call": response["is_business_call"],
            "products_mentioned": response["products_mentioned"],
            "confidence_score": response["confidence_score"],
        }

        results.append(combined)
        
    return results, failed


