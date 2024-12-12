import requests
import argparse
import pandas as pd


def get_document_segments(dataset_id, document_id, excel_name="output"):
    # Define the URL
    url = f"http://36.213.66.106:8081/v1/datasets/{dataset_id}/documents/{document_id}/segments"

    # Define the headers
    api_key = "dataset-1micH4uC6rLSs5WuScF8xmSl"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        rows = []
        # Parse the JSON response
        data = response.json()["data"]
        for segment in data:
            Content1 = segment["content"] + "\n" + ",".join(segment["keywords"])
            Answer = segment["answer"]
            Content2 = segment["content"] + "\n" + ",".join(segment["keywords"])
            rows.append([Content1, Answer, Content2])
        df = pd.DataFrame(rows)
        df.to_excel(f"no_git_oic/{excel_name}.xlsx", index=False, header=False)
    else:
        # Print the error message
        print(f"Error: {response.status_code} - {response.text}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch document segments from the API."
    )
    parser.add_argument("--dataset_id", required=True, help="ID of the dataset")
    parser.add_argument("--document_id", required=True, help="ID of the document")
    parser.add_argument(
        "--excel_name",
        default="火炬电子2023年年度报告 2024-03-19",
        help="name of the excel",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    get_document_segments(args.dataset_id, args.document_id, args.excel_name)


if __name__ == "__main__":
    # Define the API key, dataset ID, and document ID
    """
    python test/usua2/get_dify_doc_out.py --dataset_id 0a547253-9678-4050-98eb-84506170c35f \
        --document_id 855b1e2f-f688-42f0-a9c1-876a192b7765 \
        --excel_name "火炬电子2023年年度报告 2024-03-19"
        
    868be66e-a8a6-48b5-8618-38917e602334
    c0a310de-4ae2-45a3-8672-04ee4c67c0b2
    """

    main()
