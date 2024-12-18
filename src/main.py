import json
import os
import shutil
import pandas as pd
from huggingface_hub import InferenceClient
from src.helpers import load_transcripts_to_df
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN is not set in the environment. Please check your .env file."
    )


def run_llm_inference(df: pd.DataFrame, error_folder: str = "../error") -> pd.DataFrame:
    """
    Runs inference using HuggingFace LLM on the raw text and updates sentiment, follow-up_needed, and tag columns.
    Parameters:
        df (pd.DataFrame): DataFrame containing the transcript data.
    Returns:
        pd.DataFrame: Updated DataFrame with inference results.
    """
    client = InferenceClient(api_key=HF_TOKEN)

    os.makedirs(error_folder, exist_ok=True)

    error_count = 0

    for idx, row in df.iterrows():
        messages = [
            {
                "role": "system",
                "content": 'You are expert customer service agent. You are to decide the customer sentiment in their conversation with a PA agent. The sentiment can only be one of three categories - positive, negative, neutral. You are also to decide if the issue is resolved or if a follow-up action is needed.  Please also tag the conversation with one of three tags - either "claim" if the call is about a claim, "policy" if it is a question about the insurance policy, "tech" if it is about not being able to log in to online services, "pre-auth" if a member wants a treatment authorised, or "other" if the call cannot be categorised. Please provide the answer in dictionary format eg {"sentiment": "neutral", "follow_up_needed": "yes", "tag": "claim"}. Please only provide the dictionary response and no explanation.',
            },
            {
                "role": "user",
                "content": f"Here is the conversation transcript:\n{row['raw']}",
            },
        ]

        try:
            stream = client.chat.completions.create(
                model="mistralai/Mistral-Nemo-Instruct-2407",
                messages=messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=0.7,
                stream=True,
            )
            response_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content

            # Parse as JSON
            try:
                result = json.loads(response_content)
                df.at[idx, "sentiment"] = result.get("sentiment", "")
                df.at[idx, "follow_up_needed"] = result.get("follow_up_needed", "")
                df.at[idx, "tag"] = result.get("tag", "")
            except json.JSONDecodeError:
                transcript_name = row["name"]
                shutil.move(
                    os.path.join("../data", transcript_name),
                    os.path.join(error_folder, transcript_name),
                )
                print(
                    f"Invalid JSON response for row {idx} - move ({transcript_name}) to error folder for further investigation"
                )
                error_count += 1
                continue
        except Exception as e:
            print(f"Error running inference on row {idx}: {e}")
            error_count += 1

    print(f"Total errors encountered: {error_count}")

    return df


if __name__ == "__main__":

    data_folder = "../data"
    output_folder = "../output"
    error_folder = "..error"
    os.makedirs(output_folder, exist_ok=True)

    print("Loading transcripts...")
    df_transcripts = load_transcripts_to_df(data_folder)

    print("Running LLM inference...")
    df_transcripts = run_llm_inference(df_transcripts)

    output_file = os.path.join(output_folder, "transcripts_with_inference.csv")
    df_transcripts.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")
