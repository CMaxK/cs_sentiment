from src.main import run_llm_inference
import pandas as pd
import json


def test_run_llm_inference(mocker, tmp_path):
    """
    Test that run_llm_inference updates the DataFrame correctly.
    """
    # Sample data
    data = {
        "name": ["transcript_0.txt"],
        "raw": ["\n\nPA Agent: Welcome.\n\nMember: Thank you."],
        "sentiment": [""],
        "follow_up_needed": [""],
        "tag": [""],
    }
    df = pd.DataFrame(data)

    # Mock InferenceClient
    mock_client = mocker.patch("src.main.InferenceClient")
    mock_stream = [
        mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    delta=mocker.MagicMock(
                        content=json.dumps(
                            {
                                "sentiment": "positive",
                                "follow_up_needed": "yes",
                                "tag": "claim",
                            }
                        )
                    )
                )
            ]
        ),
    ]
    mock_client.return_value.chat.completions.create.return_value = mock_stream

    # Temp error folder
    error_folder = tmp_path / "error"
    error_folder.mkdir()

    updated_df = run_llm_inference(df, str(error_folder))

    assert (
        updated_df.loc[0, "sentiment"] == "positive"
    ), f"Expected 'positive', got '{updated_df.loc[0, 'sentiment']}'"
    assert (
        updated_df.loc[0, "follow_up_needed"] == "yes"
    ), f"Expected 'yes', got '{updated_df.loc[0, 'follow_up_needed']}'"
    assert (
        updated_df.loc[0, "tag"] == "claim"
    ), f"Expected 'claim', got '{updated_df.loc[0, 'tag']}'"


def test_model_accuracy():
    """
    Test to ensure the model accuracy meets the expected threshold.
    """

    model_predictions = pd.read_excel("notebooks/llm_validation.xlsx")
    ground_truth = pd.read_excel("notebooks/llm_validation_labelled.xlsx")

    # Ensure both DataFrames have the same length
    assert len(model_predictions) == len(
        ground_truth
    ), "Mismatch in number of rows between predictions and ground truth."

    # Columns to check
    columns_to_check = ["sentiment", "follow_up_needed", "tag"]
    accuracy_threshold = 85  # Minimum acceptable accuracy (%)

    accuracies = {}
    for col in columns_to_check:
        correct_predictions = (model_predictions[col] == ground_truth[col]).sum()
        accuracy = (correct_predictions / len(model_predictions)) * 100
        accuracies[col] = accuracy

        assert (
            accuracy >= accuracy_threshold
        ), f"Accuracy for {col} is below threshold: {accuracy:.2f}%"

    # Calculate overall accuracy
    total_correct = sum(
        (model_predictions[col] == ground_truth[col]).sum() for col in columns_to_check
    )
    overall_accuracy = (
        total_correct / (len(model_predictions) * len(columns_to_check))
    ) * 100

    assert (
        overall_accuracy >= accuracy_threshold
    ), f"Overall accuracy is below threshold: {overall_accuracy:.2f}%"
