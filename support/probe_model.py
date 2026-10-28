#!/usr/bin/env python3
# probe_models.py
# Step 1 of the multi-model sweep: find out which models your project can
# actually call, before committing thousands of requests to one that cannot.
#
# One tiny request per candidate. Reports PASS, or the specific reason it failed
# (not enabled, wrong region, no quota, no permission, bad ID).
#
# Colab:
#     %run probe_models.py

import os
import json
import time
import traceback

from google.colab import userdata          # noqa
from google.oauth2 import service_account
from google import genai
from google.genai import types

LOCATION = 'us-central1'
ALT_LOCATIONS = ['us-east5', 'europe-west4']   # some MaaS models are region-locked

info = json.loads(__import__('os').environ.get('GCP_SA_JSON'))
CREDS = service_account.Credentials.from_service_account_info(
    info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
PROJECT = info['project_id']
# Credential diagnostics omitted from the public source.
def make_client(loc):
    return genai.Client(vertexai=True, project=PROJECT, location=loc,
                        credentials=CREDS)


# ---------------------------------------------------------------------------
# Candidates. These are IDs to PROBE, not IDs I am asserting exist. Model Garden
# names and regional availability change; the probe is the source of truth.
# Grouped so you can pick 2 frontier + 2 open weight.
# ---------------------------------------------------------------------------

CANDIDATES = [
    # --- frontier, first party ---
    ('frontier', 'gemini-2.5-flash'),                 # your anchor, known good
    ('frontier', 'gemini-2.5-flash-lite'),
    ('frontier', 'gemini-2.5-pro'),                   # your judge
    ('frontier', 'gemini-3.1-pro'),
    ('frontier', 'gemini-3-flash'),

    # --- open weight, served as MaaS (no endpoint to deploy, pay per token) ---
    ('open', 'publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas'),
    ('open', 'publishers/meta/models/llama-4-scout-17b-16e-instruct-maas'),
    ('open', 'publishers/meta/models/llama-3.3-70b-instruct-maas'),
    ('open', 'publishers/qwen/models/qwen3-235b-a22b-instruct-2507-maas'),
    ('open', 'publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas'),
    ('open', 'publishers/moonshotai/models/kimi-k2-instruct-maas'),
    ('open', 'publishers/moonshotai/models/kimi-k2-thinking-maas'),
    ('open', 'publishers/deepseek-ai/models/deepseek-v3.1-maas'),
    ('open', 'publishers/openai/models/gpt-oss-120b-maas'),

    # --- Gemma. Usually NOT serverless on Vertex: you deploy it to an endpoint
    #     yourself and pay for the GPU by the hour. If these fail, that is why,
    #     and Gemma is a poor fit for a free-trial budget. ---
    ('open', 'gemma-3-27b-it'),
    ('open', 'publishers/google/models/gemma-3-27b-it'),
]

PROMPT = 'Reply with exactly one word: OK'
CFG = types.GenerateContentConfig(max_output_tokens=16, temperature=0.0)


def classify(e):
    s = str(e)
    low = s.lower()
    # AUTH must be tested FIRST. "invalid_grant: account not found" is the
    # suspended-project error and it contains the words "not found", so a
    # naive ordering reports it as a bad model ID and sends you hunting for
    # the wrong problem. That exact string is what a suspension looks like.
    if ('invalid_grant' in low or 'unauthorized' in low or '401' in s
            or 'account not found' in low):
        return 'AUTH', 'credentials rejected, stop and fix this first'
    if '429' in s or 'quota' in low or 'resource_exhausted' in low:
        return 'NO QUOTA', 'enabled but zero quota granted, request an increase'
    if '403' in s or 'permission' in low or 'denied' in low:
        return 'NO ACCESS', 'model not enabled in Model Garden, or SA lacks the role'
    if '400' in s and 'billing' in low:
        return 'BILLING', 'billing not linked'
    if '404' in s or 'not found' in low or 'not supported' in low:
        return 'NOT FOUND', 'wrong id, or not available in this region'
    return 'ERROR', s.split('\n')[0][:90]


def probe(client, model):
    t0 = time.time()
    r = client.models.generate_content(model=model, contents=PROMPT, config=CFG)
    dt = time.time() - t0
    txt = (r.text or '').strip().replace('\n', ' ')
    return dt, txt


def main():
    print('\nregion:', LOCATION)
    print('probing %d candidates, one tiny request each\n' % len(CANDIDATES))

    client = make_client(LOCATION)
    rows = []
    for tier, model in CANDIDATES:
        short = model.split('/')[-1]
        try:
            dt, txt = probe(client, model)
            rows.append((tier, model, 'PASS', '%.1fs' % dt, txt[:40]))
            print('  PASS      %-8s %-52s %5.1fs  %r' % (tier, short, dt, txt[:30]))
        except Exception as e:
            code, why = classify(e)
            rows.append((tier, model, code, '', why))
            print('  %-9s %-8s %-52s %s' % (code, tier, short, why))

    # Retry every NOT FOUND in the alternate regions. Region-locked MaaS models
    # are the single most common false negative here.
    retry = [(t, m) for t, m, c, _, _ in rows if c == 'NOT FOUND']
    if retry:
        for loc in ALT_LOCATIONS:
            print('\nretrying %d not-found models in %s' % (len(retry), loc))
            c2 = make_client(loc)
            still = []
            for tier, model in retry:
                short = model.split('/')[-1]
                try:
                    dt, txt = probe(c2, model)
                    print('  PASS      %-8s %-52s %5.1fs  (%s)' % (tier, short, dt, loc))
                    rows = [r for r in rows if r[1] != model]
                    rows.append((tier, model, 'PASS', '%.1fs' % dt, 'region=' + loc))
                except Exception as e:
                    still.append((tier, model))
            retry = still
            if not retry:
                break

    print('\n' + '=' * 74)
    ok = [r for r in rows if r[2] == 'PASS']
    print('CALLABLE: %d of %d' % (len(ok), len(rows)))
    for tier in ('frontier', 'open'):
        sel = [r for r in ok if r[0] == tier]
        print('\n  %s (%d):' % (tier, len(sel)))
        for r in sel:
            print('     %-58s %s' % (r[1], r[4]))
    print('=' * 74)

    with open('model_probe.json', 'w') as f:
        json.dump([{'tier': t, 'model': m, 'status': c, 'latency': l, 'note': n}
                   for t, m, c, l, n in rows], f, indent=2)
    print('\nwrote model_probe.json')
    print('Paste this output back. Pick 2 frontier + 2 open from the CALLABLE list.')


if __name__ == '__main__':
    main()