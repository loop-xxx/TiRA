import argparse
import csv
import re
import os
import json
import sys
from glob import glob

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from fraction import Fraction

load_dotenv('.env')

MAX_INT = sys.maxsize

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str,
                    help='Sub-dataset variant: "easy" for gsm8k-easy, "challenge" for gsm8k-challenge. Omit for base gsm8k.', required=True)
args = parser.parse_args()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def parse_answer_number(number_text):
    if '/' in number_text:
        denominator = number_text.split('/')[1]
        numerator = number_text.split('/')[0]
        if is_number(denominator) and is_number(numerator):
            if denominator == '0':
                return round(float(numerator.replace(',', '')))
            frac = Fraction(number_text.replace(',', ''))
            return round(float(frac.numerator / frac.denominator))
        return None
    if float(number_text.replace(',', '')) == float('inf'):
        return None
    return round(float(number_text.replace(',', '')))


def extract_answer_number(completion):
    text = completion.split('####')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            return parse_answer_number(match.group())
        else:
            return None
    else:
        return None


def extract_last_answer_number(completion):
    matches = re.findall(r'[\-+]?\d*[\.,/]?\d+', completion)
    for match in reversed(matches):
        value = parse_answer_number(match)
        if value is not None:
            return value
    return None


def extract_r_value(folder_name: str) -> int:
    """Extract effective rank from folder name for sorting.

    LoRA folders use ...-r=128-..., while TiRA folders use ...-M=32-K=32-...
    and should be grouped by the equivalent rank K * M.
    """
    method_match = re.search(r'-(tira)-', folder_name, flags=re.IGNORECASE)
    if method_match:
        mk = re.search(r'M=(\d+)-K=(\d+)', folder_name, flags=re.IGNORECASE)
        if mk:
            m_val, k_val = int(mk.group(1)), int(mk.group(2))
            return k_val * m_val
        return MAX_INT

    match = re.search(r'(?:^|-)r=(\d+)(?:-|$)', folder_name)
    return int(match.group(1)) if match else MAX_INT


def extract_peft_type(folder_name: str) -> str:
    """Extract PEFT method from folder name."""
    match = re.search(r'-(lora|tira)-', folder_name, flags=re.IGNORECASE)
    return match.group(1).lower() if match else 'unknown'


def compute_accuracy(jsonl_path, answer_json):
    """Compare predictions in jsonl_path against ground-truth answers in answer_json."""
    json_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json_data.append(json.loads(line))
            except Exception as e:
                print(e)
    contents = [line for line in json_data if isinstance(line, dict) and 'context' in line]
    if len(contents) != len(answer_json):
        print(f'Warning: pred count {len(contents)} != gt count {len(answer_json)} for {jsonl_path}')
    correct = []
    for _answer, content in zip(answer_json, contents):
        pred_num = extract_last_answer_number(content['pred'])
        prompt_pred_num = extract_answer_number(content['pred'])
        ans_num = float(_answer['output'].split('####')[-1].strip().replace(',', ''))
        correct.append(pred_num == ans_num or prompt_pred_num == ans_num)
    return float(np.asarray(correct).mean()) if correct else 0.0


eval_paths = glob('<fill your result dir here>')
eval_task = f'gsm8k-{args.level}'
test_json_path = f'data_file/llm_adapt/gsm8k/{args.level}/test.json'

with open(test_json_path, 'r', encoding='utf-8') as f:
    answer_json = json.load(f)

all_data = {}

for eval_path in eval_paths:
    jsonl_files = glob(f'{eval_path}/*{eval_task}*/checkpoint-*/trainer_state.json')
    if not jsonl_files:
        print(f'No trainer_state.json files found in {eval_path} for task {eval_task}')
        continue

    csv_header = ['peft', 'r', 'eval_loss', 'acc']
    csv_data = []

    for json_file in tqdm(jsonl_files):
        print(json_file)
        peft_type = extract_peft_type(json_file)
        r_value = extract_r_value(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        run_dir = os.path.dirname(os.path.dirname(json_file))
        pred_jsonls = glob(f'{run_dir}/*.jsonl')
        if pred_jsonls:
            acc = compute_accuracy(pred_jsonls[0], answer_json) * 100.0
        else:
            print(f'No prediction jsonl found in {run_dir}')
            acc = ''

        csv_row = [peft_type, r_value, data['best_metric'], acc]
        csv_data.append(csv_row)

    out_csv = f'{eval_path}/results_{eval_task}.csv'
    csv_data.sort(key=lambda row: (row[0], row[1]))
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
    print(f'Saved: {out_csv}')
