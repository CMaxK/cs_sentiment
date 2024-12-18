import os
import pandas as pd


def preprocess_text(raw_text: str) -> str:
    """
    Preprocess the raw text by adding line breaks before speaker identifiers.
    Parameters:
        raw_text (str): The raw conversation transcript.
    Returns:
        str: Formatted text with proper line breaks.
    """
    formatted_text = raw_text.strip()
    formatted_text = (
        formatted_text.replace("PA Agent:", "\n\nPA Agent:")
        .replace("Member:", "\n\nMember:")
        .strip()
    )
    if not formatted_text.startswith("\n"):
        formatted_text = "\n" + formatted_text
    return formatted_text


def load_transcripts_to_df(data_folder: str) -> pd.DataFrame:
    """
    Reads all transcript_x.txt files from a specified folder into a DataFrame.
    Parameters:
        data_folder (str): Path to the folder containing transcript files.
    Returns:
        pd.DataFrame: A DataFrame containing file name, raw text, and empty columns for sentiment, follow-up_needed, and tag.
    """
    data = []

    for file_name in os.listdir(data_folder):
        if file_name.startswith("transcript_") and file_name.endswith(".txt"):
            file_path = os.path.join(data_folder, file_name)

            try:
                # Read file
                with open(file_path, "r", encoding="utf-8") as file:
                    raw_text = file.read().strip()
                    formatted_text = preprocess_text(raw_text)

                # Append file
                data.append(
                    {
                        "name": file_name,
                        "raw": formatted_text,
                        "sentiment": "",
                        "follow_up_needed": "",
                        "tag": "",
                    }
                )
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    df = pd.DataFrame(
        data, columns=["name", "raw", "sentiment", "follow_up_needed", "tag"]
    )
    return df
