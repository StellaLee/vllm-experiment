#!/usr/bin/env python3
"""Convert a WildChat-1M parquet shard into the exact ShareGPT-compatible JSON format
src/replay_sharegpt.py's --dataset mode expects: a list of {id, conversations: [{from, value}]}.
WildChat uses role=user/assistant (mapped to human/gpt) and includes real per-turn moderation
flags (toxic/redacted) this project doesn't need for a load-testing content source -- filtered
out for cleanliness, not because they'd break anything downstream.

Usage: convert_wildchat.py IN_PARQUET OUT_JSON [--lang English] [--max-convs N]
"""
import argparse
import json
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_parquet')
    ap.add_argument('out_json')
    ap.add_argument('--lang', default='English')
    ap.add_argument('--max-convs', type=int, default=20000)
    args = ap.parse_args()

    t = pq.read_table(args.in_parquet, columns=['conversation_hash', 'language', 'conversation'])
    rows = t.to_pylist()

    out = []
    skipped_lang, skipped_toxic, skipped_short = 0, 0, 0
    for row in rows:
        if row['language'] != args.lang:
            skipped_lang += 1
            continue
        conv = row['conversation']
        if any(turn.get('toxic') or turn.get('redacted') for turn in conv):
            skipped_toxic += 1
            continue
        if len(conv) < 2:
            skipped_short += 1
            continue
        conversations = []
        for turn in conv:
            role = turn['role']
            frm = 'human' if role == 'user' else 'gpt' if role == 'assistant' else None
            if frm is None:
                continue
            conversations.append({'from': frm, 'value': turn['content']})
        if len(conversations) < 2:
            skipped_short += 1
            continue
        out.append({'id': row['conversation_hash'], 'conversations': conversations})
        if len(out) >= args.max_convs:
            break

    with open(args.out_json, 'w') as f:
        json.dump(out, f)

    print(f'[wildchat] kept={len(out)} skipped_lang={skipped_lang} skipped_toxic={skipped_toxic} skipped_short={skipped_short}')
    lens = [len(c['conversations']) for c in out]
    print(f'[wildchat] turn-count dist: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}')


if __name__ == '__main__':
    main()
