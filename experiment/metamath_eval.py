import csv
import re
from glob import glob
import jsonlines
import numpy as np
from dotenv import load_dotenv
import evaluate
import os
import json

from tqdm import tqdm

load_dotenv('.env')


import json
import re
from fraction import Fraction
import sys

MAX_INT = sys.maxsize


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


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def extract_answer(dataset, sentence: str) -> float:
    if dataset in ('gsm8k', 'meta_math'):
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'\d+', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]


eval_paths = glob('<fill your result dir here>')
eval_task = 'meta_math'


eval_task_json_map = {
    'meta_math': 'data_file/meta_math/test.json'
}


def extract_method_and_hyperparams(folder_name, source_path=None):
    """Extract (model, method, hparam_str, sort_key) from folder name.
    Folder format: {output}/{model}-{dataset}-{peft_type}-lr=...-(M=..-K=..|r=..)-...
    For tira: hparams = M=...-K=... ; otherwise hparams = r=... .
    sort_key sorts by (model, method, then numeric hparams).
    """
    search_target = source_path if source_path is not None else folder_name

    model_match = re.search(rf'[\\/]([^\\/]+?)-{eval_task}-', search_target)
    model = model_match.group(1) if model_match else 'unknown'

    method_match = re.search(rf'-{eval_task}-([A-Za-z_]+?)-', search_target)
    method = method_match.group(1) if method_match else 'unknown'

    if method == 'tira':
        mk = re.search(r'M=(\d+)-K=(\d+)', folder_name)
        if mk:
            m_val, k_val = int(mk.group(1)), int(mk.group(2))
            return model, method, f'M={m_val}-K={k_val}', (model, method, k_val, m_val)
        return model, method, '', (model, method, 0, 0)

    r_match = re.search(r'(?<![A-Za-z])r=(\d+)', folder_name)
    if r_match:
        r_val = int(r_match.group(1))
        return model, method, f'r={r_val}', (model, method, r_val, 0)
    return model, method, '', (model, method, 0, 0)


all_data = {}
for eval_path in eval_paths:
    jsonl_files = glob(f'{eval_path}/*{eval_task}*/*.jsonl')
    csv_header = ['model', 'method', 'hyperparams', 'acc', 'last acc', 'prompt acc', 'config']
    csv_data = []
    with open(eval_task_json_map[eval_task], 'r') as file:
        answer_json = json.load(file)
    for json_file in tqdm(jsonl_files):
        print(json_file)
        json_data = []
        lines = []
        with open(json_file, 'r') as f:
            for line in f.readlines():
                lines.append(line)
        lines = [line for line in lines if len(line.strip()) > 0]
        for line in lines:
            try:
                json_data.append(json.loads(line))
            except Exception as e:
                print(e)

        configs = json_data[:2]
        contents = [line for line in json_data if line.__class__ is dict and 'context' in line.keys()]
        assert  len(contents) == len(answer_json), 'num of pred must equal to num of gt'
        answers = []
        last_answers = []
        prompt_answers = []
        for _answer, content in zip(answer_json, contents):
            pred = extract_answer(f'{eval_task}', content['pred'])
            gt = extract_answer(f'{eval_task}', content['gt'])
            last_correct = pred == gt
            pred_num = extract_answer_number(content['pred'])
            ans_num = float(_answer['response'].split('####')[-1].strip().replace(',', ''))
            prompt_correct = pred_num == ans_num
            last_answers.append(last_correct)
            prompt_answers.append(prompt_correct)
            answers.append(last_correct or prompt_correct)
        acc = np.asarray(answers).mean()
        acc_last = np.asarray(last_answers).mean()
        acc_prompt = np.asarray(prompt_answers).mean()
        # folder_name = json_file.split(os.sep)[2]
        # folder_name = re.sub(r'-\d\d\d\d-\d\d-\d\d.*','',folder_name)+os.sep+json_file.split(os.sep)[-1]
        folder_name = json_file.replace(eval_task, '')
        model, method, hparam_str, sort_key = extract_method_and_hyperparams(folder_name, source_path=json_file)
        csv_row = [model, method, hparam_str, acc * 100.0, acc_last * 100, acc_prompt * 100, configs]
        if folder_name in all_data.keys():
            all_data[folder_name][eval_task] = acc * 100
        else:
            all_data[folder_name] = {eval_task: acc * 100}
        csv_data.append((sort_key, csv_row))
    csv_data.sort(key=lambda x: x[0])
    csv_data = [row for _, row in csv_data]
    with open(f'{eval_path}/results_{eval_task}.csv', 'w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
