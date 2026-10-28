#!/usr/bin/env python3
# dump_methods_facts.py  (v2)
# Prints every number Methods needs, read from whatever actually exists in Drive.
#
# v2 changes: does not assume bench_config.json exists. Falls back to extracting
# DOMAINS / FAILURE_TYPES / PROMPTS / GEN_PARAMS straight out of the workflow
# notebook's source by AST, executing only the assignment statements themselves
# so no API call can fire. Prints a full inventory first, so a failure still
# tells you what you have.
#
# Colab:
#     from google.colab import drive; drive.mount('/content/drive')
#     %run dump_methods_facts.py

import os
import re
import ast
import json
import glob
import hashlib
import difflib
import collections

OUT = {}
CFG = {}


def rule(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


# ------------------------------------------------------------------ root ----
ROOTS = ['/content/drive/MyDrive', '/content/drive/My Drive',
         os.environ.get('SPARK_DRIVE', ''), '/content', '.']
DRIVE = None
for r in ROOTS:
    if r and os.path.isdir(r):
        DRIVE = r
        break

if DRIVE in (None, '.', '/content'):
    print('!! Google Drive does not look mounted.')
    print('   Run this first, in its own cell:')
    print('       from google.colab import drive')
    print('       drive.mount(\'/content/drive\')')
    print('   Continuing against %r anyway.\n' % (DRIVE or '.'))
DRIVE = DRIVE or '.'


# ------------------------------------------------------------- inventory ----
rule('0. WHAT IS ACTUALLY THERE   (root: %s)' % DRIVE)


def walk(pat, limit=40):
    hits = glob.glob(f'{DRIVE}/**/{pat}', recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    hits.sort(key=lambda p: -os.path.getsize(p))
    return hits[:limit], len(hits)


for pat in ('*.ipynb', '*.json', '*.jsonl', '*.csv'):
    hits, n = walk(pat)
    print('\n%s  (%d found)' % (pat, n))
    for h in hits:
        print('   %9.2f MB  %s' % (os.path.getsize(h) / 1e6, h))
    if not hits:
        print('   none')

NOTEBOOKS = [p for p in glob.glob(f'{DRIVE}/**/*.ipynb', recursive=True)
             if os.path.isfile(p)]
WORKFLOW = None
for p in NOTEBOOKS:
    if 'tool_failure_benchmark_workflow' in os.path.basename(p):
        WORKFLOW = p
        break
if WORKFLOW is None and NOTEBOOKS:
    scored = []
    for p in NOTEBOOKS:
        b = os.path.basename(p).lower()
        scored.append((('workflow' in b) * 2 + ('benchmark' in b) + ('tool' in b),
                       os.path.getsize(p), p))
    scored.sort(reverse=True)
    WORKFLOW = scored[0][2]
print('\nworkflow notebook ->', WORKFLOW)


# ---------------------------------------------------------------- config ----
rule('1. FROZEN CONFIG')

WANT = ('DOMAINS', 'FAILURE_TYPES', 'PROMPTS', 'GEN_PARAMS',
        'JUDGE_MODEL', 'GEN_MODEL', 'MODEL', 'ANCHOR_MODEL',
        'N_PER_CELL', 'N_REPEATS', 'REPEATS', 'TEMPERATURE')

cands = [p for p in glob.glob(f'{DRIVE}/**/bench_config.json', recursive=True)
         if os.path.isfile(p)]

SOURCE = None
RAW = None

if cands:
    CONFIG_PATH = sorted(cands, key=len)[0]
    RAW = open(CONFIG_PATH, 'rb').read()
    CFG = json.loads(RAW)
    SOURCE = CONFIG_PATH
    print('source: bench_config.json')
    print('path  :', CONFIG_PATH)
elif WORKFLOW:
    print('no bench_config.json anywhere. Falling back to the notebook source.')
    print('source:', WORKFLOW)
    nb = json.load(open(WORKFLOW, encoding='utf-8'))
    ns = {}
    got = []

    # Only these calls may appear on the right-hand side. Anything else (an API
    # client, a file read, a model invocation) means the statement is skipped.
    SAFE_CALLS = {'dict', 'list', 'set', 'tuple', 'frozenset', 'str', 'int',
                  'float', 'bool', 'range', 'len', 'sorted', 'reversed',
                  'enumerate', 'zip', 'map', 'filter', 'sum', 'min', 'max',
                  'abs', 'round', 'repr', 'format', 'join', 'strip', 'split',
                  'replace', 'lower', 'upper', 'items', 'keys', 'values'}

    def is_safe(node):
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Await, ast.Yield, ast.YieldFrom)):
                return False
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Name) and f.id in SAFE_CALLS:
                    continue
                if isinstance(f, ast.Attribute) and f.attr in SAFE_CALLS:
                    continue
                return False
        return True

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        src = '\n'.join(l for l in src.split('\n')
                        if not l.lstrip().startswith(('!', '%')))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # Execute every SAFE top-level assignment, not only the wanted ones, so
        # that helper constants a wanted value refers to are in scope. Bare
        # expressions and any call-bearing assignment are never executed.
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not is_safe(node.value):
                if any(n in WANT for n in names):
                    print('   skipped %s: right-hand side calls something' % names)
                continue
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             '<cfg>', 'exec'), ns, ns)
                for n in names:
                    if n in WANT and n in ns:
                        got.append(n)
            except Exception as e:
                if any(n in WANT for n in names):
                    print('   could not evaluate %s: %s: %s'
                          % (names, type(e).__name__, e))
    for k in WANT:
        if k in ns:
            CFG[k] = ns[k]
    SOURCE = WORKFLOW
    print('extracted:', sorted(set(got)))
    RAW = json.dumps(CFG, sort_keys=True, ensure_ascii=False,
                     default=str).encode()
else:
    print('Nothing to read. Mount Drive and re-run.')

OUT['config_source'] = SOURCE

if CFG:
    fp_canon = hashlib.sha256(json.dumps(
        CFG, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
    print('config canon sha256[:16] =', fp_canon)
    OUT['config_fp_canon'] = fp_canon
    if cands:
        fp_raw = hashlib.sha256(RAW).hexdigest()[:16]
        print('config raw   sha256[:16] =', fp_raw)
        OUT['config_fp_raw'] = fp_raw
    else:
        print('config raw   sha256[:16] = n/a (reconstructed from notebook)')

    DOMAINS = CFG.get('DOMAINS') or []
    FAILURE_TYPES = CFG.get('FAILURE_TYPES') or {}
    PROMPTS = CFG.get('PROMPTS') or {}
    GEN = CFG.get('GEN_PARAMS') or {}

    print('\nGEN_PARAMS:', json.dumps(GEN, sort_keys=True, default=str))
    for k in ('JUDGE_MODEL', 'GEN_MODEL', 'MODEL', 'ANCHOR_MODEL',
              'N_PER_CELL', 'N_REPEATS', 'REPEATS', 'TEMPERATURE'):
        if k in CFG:
            print('%-14s = %s' % (k, CFG[k]))
            OUT[k] = CFG[k]

    dnames = list(DOMAINS)
    print('\n%d DOMAINS (paste verbatim into Methods):' % len(dnames))
    for i, d in enumerate(dnames):
        print('  %2d. %s' % (i + 1, d))

    fnames = list(FAILURE_TYPES)
    print('\n%d FAILURE TYPES:' % len(fnames))
    for i, f in enumerate(fnames):
        body = FAILURE_TYPES[f] if isinstance(FAILURE_TYPES, dict) else ''
        b = str(body).replace('\n', ' ')
        print('  %d. %-30s %s' % (i + 1, f, b[:66] + ('...' if len(b) > 66 else '')))

    OUT['domains'] = dnames
    OUT['failure_types'] = fnames
    OUT['gen_params'] = {k: str(v) for k, v in GEN.items()}

    # ------------------------------------------------------------ prompts --
    rule('2. THE FOUR CONDITION PROMPTS, VERBATIM')
    OUT['prompts'] = {}
    for c in ['neutral', 'deploy', 'deploy_strict', 'deploy_trustcheck']:
        if c not in PROMPTS:
            print('\nMISSING:', c)
            continue
        p = PROMPTS[c]
        h = hashlib.sha256(p.encode()).hexdigest()[:12]
        print('\n--- %s   (sha256[:12] %s, %d chars) ---' % (c, h, len(p)))
        print(p)
        OUT['prompts'][c] = {'text': p, 'sha256_12': h, 'chars': len(p)}

    print('\n--- pairwise diffs, for the ablation table ---')
    for a, b in [('neutral', 'deploy'), ('deploy', 'deploy_strict'),
                 ('deploy', 'deploy_trustcheck')]:
        if a in PROMPTS and b in PROMPTS:
            d = [l for l in difflib.unified_diff(
                PROMPTS[a].split('\n'), PROMPTS[b].split('\n'), lineterm='', n=0)
                if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
            print('\n%s -> %s  (%d changed lines)' % (a, b, len(d)))
            for l in d[:14]:
                print('   ', l[:110])


# ------------------------------------------------------- realized grid ------
rule('3. THE REALIZED GRID, COUNTED FROM THE DATA')

res = [p for p in glob.glob(f'{DRIVE}/**/*.jsonl', recursive=True)
       if os.path.isfile(p)]
res.sort(key=lambda p: -os.path.getsize(p))

if not res:
    print('no .jsonl found')
else:
    for p in res[:8]:
        print('  %9.2f MB  %s' % (os.path.getsize(p) / 1e6, p))

    MASTER = res[0]
    print('\nusing the largest:', MASTER)

    grid = collections.Counter()
    ftype_by_rep = collections.defaultdict(set)
    keys_seen = collections.Counter()
    n = bad = 0
    with open(MASTER, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            n += 1
            if n == 1:
                keys_seen.update(r.keys())
            grid[(r.get('condition'), r.get('repeat_index'))] += 1
            ft = r.get('failure_type')
            if ft is not None:
                ftype_by_rep[r.get('repeat_index')].add(ft)

    print('\nrows: %d parsed, %d unparseable' % (n, bad))
    print('fields on row 1:', sorted(keys_seen))

    conds = sorted({k[0] for k in grid if k[0] is not None})
    reps = sorted({k[1] for k in grid if k[1] is not None})
    if conds:
        hdr = '%-26s' % 'condition' + ''.join('%9s' % ('rep %s' % r) for r in reps) + '%10s' % 'total'
        print('\n' + hdr)
        per_cond = {}
        for c in conds:
            row = [grid[(c, r)] for r in reps]
            per_cond[c] = sum(row)
            print('%-26s' % c + ''.join('%9d' % v for v in row) + '%10d' % sum(row))
        OUT['per_condition_total'] = per_cond
        OUT['grid'] = {'%s|%s' % (k[0], k[1]): v for k, v in grid.items()}
    else:
        print('\nno "condition" field on these rows. Field names above.')

    if ftype_by_rep:
        print('\nfailure types present per repeat:')
        for rep in sorted(ftype_by_rep, key=lambda x: (x is None, x)):
            print('  repeat %s: %d types' % (rep, len(ftype_by_rep[rep])))
        if 0 in ftype_by_rep and 1 in ftype_by_rep:
            dropped = sorted(ftype_by_rep[0] - ftype_by_rep[1])
            print('\n  DROPPED after repeat 0 (%d): %s' % (len(dropped), dropped))
            print('  ^ these are the "redundant" types. Name them in Methods.')
            OUT['dropped_after_repeat0'] = dropped

    OUT['total_rows'] = n
    OUT['master_path'] = MASTER


# ----------------------------------------------------------------- judge ----
rule('4. JUDGE RUBRIC')

if not WORKFLOW:
    print('no notebook to read')
else:
    src_all = json.load(open(WORKFLOW, encoding='utf-8'))
    hit = False
    for cell in src_all.get('cells', []):
        s = ''.join(cell.get('source', []))
        m = re.search(r'JUDGE_PROMPT_V2\s*=\s*r?("""|\'\'\')(.*?)\1', s, re.S)
        if m:
            jp = m.group(2)
            sha = hashlib.sha256(jp.encode()).hexdigest()[:12]
            print('JUDGE_PROMPT_V2: %d chars, sha256[:12] = %s' % (len(jp), sha))
            labels = sorted(set(re.findall(
                r'\b(HONEST_REPORT|FABRICATION|UNFAITHFUL_REFUSAL|HEDGED)\b', jp)))
            print('labels:', labels)
            OUT['judge_prompt_sha256_12'] = sha
            OUT['judge_labels'] = labels
            OUT['judge_prompt_chars'] = len(jp)
            hit = True
            break
    if not hit:
        print('JUDGE_PROMPT_V2 not found in', WORKFLOW)


# ------------------------------------------------------------------ save ----
with open('methods_facts.json', 'w', encoding='utf-8') as fh:
    json.dump(OUT, fh, indent=2, ensure_ascii=False, default=str)

rule('DONE')
print('wrote methods_facts.json')
print('Paste this whole output back.')