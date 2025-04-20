from vqa_feedback import generate_qid_data_from_prompt
from util.data_process import read_only_prompts_from_jsonl
import argparse
import sys

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate qid_data.json")
    parser.add_argument("--input_file", type=str, default="asset/aaa.jsonl", help="Input file path")
    parser.add_argument("--output_file", type=str, default="qid_data.json", help="Output file path")
    args = parser.parse_args()
    prompts_file = args.input_file
    output_file = args.output_file
    prompts = read_only_prompts_from_jsonl(prompts_file)

    for i in range(len(prompts)):
        generate_qid_data_from_prompt(prompts[i], output_file)

if __name__ == "__main__":
    sys.exit(main())
