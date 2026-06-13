import json
import os
from modelscope.msdatasets import MsDataset

ds = MsDataset.load('modelscope/gsm8k', subset_name='main', split='test', trust_remote_code=True)

# Convert to match existing meta_math format (query/response/type)
data = []
for item in ds:
    data.append({
        "query": item["question"],
        "response": item["answer"],
        "type": "GSM8K"
    })

output_path = os.path.join(os.path.dirname(__file__), 'test.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(data)} examples to {output_path}")
