import os
import json




def load_meeting(path):
    transcript_path = os.path.join(path, "transcript.json")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found at {transcript_path}")

    meeting_path = os.path.join(path, "meeting-info.json")
    if not os.path.exists(meeting_path):
        raise FileNotFoundError(f"Meeting info file not found at {meeting_path}")

    summary_path = os.path.join(path, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    with open(meeting_path,"r") as f:
        meeting = json.load(f)

    if not transcript or not transcript["data"] or not meeting:
        raise ValueError(f"Transcript or meeting info is empty")

    transcript_data = transcript["data"]
    full_text = ""
    for s in transcript_data:
        full_text += s["speaker_name"] + ": " + s["sentence"] + "\n"

    call_type_derived = None
    if meeting:
        emails = meeting["allEmails"]
        domains = set(e.split("@")[1] for e in emails if "@" in e)
        call_type_derived = "internal" if len(domains) == 1 else "external"
        title = (meeting["title"] or "").lower()
        if "support case" in title or "case #" in title:
            call_type_derived = "support"
    # ─────────────────────────────────────────────────────────────────────────

    # ── Change 2: churn signals from keyMoments in summary.json ──────────────
    has_churn_signal = False
    has_concern = False
    if summary:
        key_moments = summary.get("keyMoments", [])
        has_churn_signal = any(km["type"] == "churn_signal" for km in key_moments)
        has_concern = any(km["type"] == "concern" for km in key_moments)

    return {
        "folder": os.path.basename(path),
        "meeting_id": meeting["meetingId"],
        "title":      meeting["title"],
        "host":       meeting["host"],
        "start_time": meeting["startTime"],
        "end_time":   meeting["endTime"],
        "duration":   meeting["duration"],
        "attendees":  meeting["invitees"],
        "sentence_count": len(transcript_data),
        "call_type_derived": call_type_derived,
        "has_churn_signal": has_churn_signal,
        "has_concern": has_concern,
        "ref_sentiment_label": summary["overallSentiment"] if summary else None,
        "ref_sentiment_score": summary["sentimentScore"]   if summary else None,

        "transcript": full_text,
    } 


def load_all_meetings(dir):

    meetings = []
    failed = []
    for folder in os.listdir(dir):
        if not os.path.isdir(os.path.join(dir, folder)):
            continue

        try:
            meeting = load_meeting(os.path.join(dir, folder))
            meetings.append(meeting)
        except (FileNotFoundError, ValueError):
            failed.append(folder)

    return meetings, failed
