"""Print a Markdown timing table from checked reports."""
import argparse
import json
from pathlib import Path
from benchmarks.render import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--repeat', type=Path)
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding='utf-8'))
    after = json.loads(args.after.read_text(encoding='utf-8'))
    repeat = json.loads(args.repeat.read_text(encoding='utf-8')) if args.repeat else {'results': []}
    initial = {item['case']: item for item in before['results']}
    repeated = {item['case']: item for item in repeat['results']}
    print('| Case | Before median / p95 (ms) | After median / p95 (ms) | Median speedup | Repeat median (ms) |')
    print('| --- | ---: | ---: | ---: | ---: |')
    for item in after['results']:
        compare(item, before)
        old = initial[item['case']]
        check = repeated.get(item['case'])
        if check:
            compare(check, before)
        repeated_ms = f"{check['median_ms']:.1f}" if check else '—'
        print(f"| {item['case']} | {old['median_ms']:.1f} / {old['p95_ms']:.1f} "
              f"| {item['median_ms']:.1f} / {item['p95_ms']:.1f} "
              f"| {item['median_speedup']:.2f}× | {repeated_ms} |")


if __name__ == '__main__':
    main()
