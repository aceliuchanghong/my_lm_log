import requests
import argparse
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_document_segments(dataset_id, document_id, excel_name="output", year="2023"):
    # Define the URL
    url = f"http://36.213.66.106:8081/v1/datasets/{dataset_id}/documents/{document_id}/segments"

    # Define the headers
    api_key = "dataset-1micH4uC6rLSs5WuScF8xmSl"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Make the GET request
    logger.info(colored(f"request start...", "green"))
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        logger.info(colored(f"recieved req success", "green"))
        year_key_word = year if len(year) > 0 else ""
        rows = []
        # Parse the JSON response
        data = response.json()["data"]
        for segment in data:
            Content1 = (
                segment["content"]
                + "\n"
                + "火炬,"
                + ",".join(segment["keywords"] + ["火炬电子"] + [year_key_word])
            )
            Answer = segment["answer"]
            Content2 = (
                segment["content"]
                + "\n"
                + "火炬,"
                + ",".join(segment["keywords"] + ["火炬电子"] + [year_key_word])
            )
            rows.append([Content1, Answer, Content2])
        df = pd.DataFrame(rows)
        df.to_excel(f"no_git_oic/{excel_name}.xlsx", index=False, header=False)
        logger.info(colored(f"result in no_git_oic/{excel_name}.xlsx", "green"))
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
    parser.add_argument("--year", default="2023", help="Year of the document")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    get_document_segments(args.dataset_id, args.document_id, args.excel_name, args.year)


if __name__ == "__main__":
    # Define the API key, dataset ID, and document ID
    # export no_proxy="localhost,36.213.66.106,127.0.0.1"
    """
    python test/usua2/get_dify_doc_out.py --dataset_id 0a547253-9678-4050-98eb-84506170c35f \
        --document_id 855b1e2f-f688-42f0-a9c1-876a192b7765 \
        --excel_name "火炬电子2023年年度报告 2024-03-19" \
        --year "2023"
    """

    main()
