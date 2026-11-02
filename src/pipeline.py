import argparse
import pandas as pd
import os
from load_data import load_all_meetings
from analyze import analyze_call

RAW_DIR_FOLDER = "data/raw"

def main():
    parser = argparse.ArgumentParser(description="Classify meeting transcripts with a local LLM.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached LLM responses and call the model again.",
    )
    args = parser.parse_args()

    # Step 1 : Load data from all meetings
    meetings, failed = load_all_meetings(RAW_DIR_FOLDER)
    print(f"Loaded {len(meetings)} meetings and {len(failed)} failed meetings")

    # Step 2 : Analyze each meeting
    results, failed = analyze_call(meetings, force=args.force)
    print(f"Analyzed {len(results)} meetings and {len(failed)} failed meetings")
    # Step 3 : Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/results.csv", index=False)
    print(f"Saved results to data/results.csv")

    
    # Step 4 : Save failed meetings to CSV
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv("data/failed.csv", index=False)



if __name__ == "__main__":
    main()