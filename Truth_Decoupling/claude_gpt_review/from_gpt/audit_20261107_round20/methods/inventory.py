"""Check literature-index completeness and local advisor-package links, read-only."""
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import unquote


def inspect(root):
    package = root / '老师审核/2026-11-06'
    literature = package / '04_相关文献与顶会'
    files = sorted('notes/' + p.name for p in (literature / 'notes').glob('*.md'))
    links = re.findall(r'\]\((notes/[^)]+)\)', (literature / 'index.md').read_text())
    duplicates = sorted(k for k, count in Counter(links).items() if count > 1)
    missing = sorted(k for k in links if not (literature / k).exists())
    unindexed = sorted(set(files) - set(links))
    broken = []
    for path in sorted(package.rglob('*.md')):
        for href in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            target = unquote(href.split('#')[0])
            if target and not re.match(r'^[a-zA-Z]+:', target) and not (path.parent / target).exists():
                broken.append({'file': str(path.relative_to(root)), 'target': href})
    result = {'base_commit': '63e3674b6d98729e453a4b259ee7724b9234711a',
              'scope': 'index/file/link integrity; not a full-text replication of all 75 papers',
              'note_files': len(files), 'index_links': len(links), 'unique_index_links': len(set(links)),
              'duplicate_links': duplicates, 'missing_notes': missing, 'unindexed_notes': unindexed,
              'broken_local_markdown_links': broken}
    if (len(files), len(links), len(set(links))) != (75, 75, 75) or duplicates or missing or unindexed or broken:
        raise ValueError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    # methods -> round20 -> from_gpt -> claude_gpt_review -> project
    root = Path(__file__).resolve().parents[4]
    out = inspect(root)
    Path(__file__).with_name('inventory.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))
