import json

def write_jsonl(path, records):
 with open(path, "w", encoding="utf-8") as f:
  for r in records:
   f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
 with open(path, "r", encoding="utf-8") as f:
  for line in f:
   yield json.loads(line)