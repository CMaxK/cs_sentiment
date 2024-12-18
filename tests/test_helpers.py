import pandas as pd
from src.helpers import preprocess_text, load_transcripts_to_df


def test_preprocess_text():
    """
    Test that preprocess_text correctly formats the input text.
    """
    raw_text = "PA Agent: Hello. Member: Hi there."
    expected_output = "\nPA Agent: Hello. \n\nMember: Hi there."
    assert preprocess_text(raw_text) == expected_output


def test_load_transcripts_to_df(tmp_path):
    """
    Test that load_transcripts_to_df loads data correctly.
    """
    # Create temporary test folder and file
    temp_folder = tmp_path / "data"
    temp_folder.mkdir()

    test_file = temp_folder / "transcript_0.txt"
    test_file.write_text("PA Agent: Welcome. Member: Thank you.")

    # Call the function
    df = load_transcripts_to_df(str(temp_folder))

    assert len(df) == 1
    assert len(df.columns) == 5
    assert df.loc[0, "name"] == "transcript_0.txt"
