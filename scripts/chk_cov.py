import json

with open('coverage.json', encoding='utf-8') as f:
    d = json.load(f)

with open('missing_coverage.txt', 'w', encoding='utf-8') as out:
    for file, info in d['files'].items():
        if info['summary']['percent_covered'] < 100.0:
            missing = info['missing_lines']
            if missing:
                out.write(f"{file}: {info['summary']['percent_covered']:.2f}% missing lines: {missing}\n")
