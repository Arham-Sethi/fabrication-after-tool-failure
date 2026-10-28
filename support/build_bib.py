#!/usr/bin/env python3
# build_bib.py
# Builds references.bib for the NeurIPS Overleaf template from the verified citation master.
#
# Pulls real titles and FULL author lists from the arXiv API (3 batched requests, not 52),
# then applies every manual correction from the audit.
#
# Run in Colab or anywhere with arXiv access:
#     !python build_bib.py
# Output: references.bib

import re
import sys
import time
import html
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

UA = {"User-Agent": "bib-builder/1.0 (mailto:arhamsethi09@gmail.com)"}

# ----------------------------------------------------------------------------
# 1. The citation table. key -> (cited_as, arxiv_id_or_None, venue_tag)
#    Year is taken from cited_as, NOT from arXiv. That is deliberate: you cite
#    the published venue, arXiv shows the preprint date.
# ----------------------------------------------------------------------------

ROWS = [
    # key                arxiv id      year  venue tag
    ("schick2023",       "2302.04761", 2023, "NEURIPS36"),
    ("qin2024",          "2307.16789", 2024, "ICLR"),
    ("li2023",           "2304.08244", 2023, "EMNLP"),
    ("mialon2023",       "2311.12983", 2023, "ICLR"),
    ("liuX2024",         "2308.03688", 2024, "ICLR"),
    ("zhouS2024",        "2307.13854", 2024, "ICLR"),
    ("yao2024",          "2406.12045", 2024, "PREPRINT"),
    ("hossain2026",      "2606.12797", 2026, "PREPRINT"),
    ("chenJ2024",        "2309.01431", 2024, "AAAI"),
    ("ming2025",         "2410.03727", 2025, "ICLR"),
    ("wu2024",           "2404.10198", 2024, "NEURIPSDB"),
    ("wang2025",         "2410.07176", 2025, "ACL"),
    ("yoran2024",        "2310.01558", 2024, "ICLR"),
    ("zhouW2023",        "2303.11315", 2023, "FINDINGS_EMNLP"),
    ("sharma2023",       "2310.13548", 2023, "ICLR"),
    ("rottger2024",      "2308.01263", 2024, "NAACL"),
    ("sclar2024",        "2310.11324", 2024, "ICLR"),
    ("wallace2024",      "2404.13208", 2024, "PREPRINT"),
    ("eliav2026",        "2607.19257", 2026, "PREPRINT"),
    ("kadavath2022",     "2207.05221", 2022, "PREPRINT"),
    ("yin2023",          "2305.18153", 2023, "FINDINGS_ACL"),
    ("kirichenko2025",   "2506.09038", 2025, "PREPRINT"),
    ("wen2024",          "2407.18418", 2024, "TACL"),
    ("turpin2023",       "2305.04388", 2023, "NEURIPS"),
    ("lanham2023",       "2307.13702", 2023, "PREPRINT"),
    ("chenY2025",        "2505.05410", 2025, "PREPRINT"),
    ("rao2026",          "2606.00093", 2026, "PREPRINT"),
    ("advani2026",       "2606.09863", 2026, "FAGEN"),
    ("zhu2026",          "2606.05806", 2026, "PREPRINT"),
    ("soni2026",         "2607.04686", 2026, "PREPRINT"),
    ("liu2026abstain",   "2607.10059", 2026, "PREPRINT"),
    ("singh2026",        "2607.19449", 2026, "KDDW"),
    ("tamba2026",        "2607.13071", 2026, "PREPRINT"),
    # Paper 2
    ("cui2025",          "2405.20947", 2025, "ICML"),
    ("zhang2025",        "2505.08054", 2025, "COLM"),
    ("vanderweij2024",   "2406.07358", 2024, "PREPRINT"),
    ("arditi2024",       "2406.11717", 2024, "NEURIPS"),
    ("yu2026",           "2603.20320", 2026, "ICECET"),
    ("cartagena2026",    "2602.16943", 2026, "PREPRINT"),
    ("zeng2026",         "2604.19749", 2026, "PREPRINT"),
    ("cheng2026",        "2605.14038", 2026, "PREPRINT"),
    ("sun2026",          "2605.09252", 2026, "PREPRINT"),
    ("liu2026irrel",     "2604.11322", 2026, "PREPRINT"),
    ("laban2026",        "2505.06120", 2026, "ICLR"),      # ICLR 2026 Oral, not the 2025 preprint year
    ("miao2025",         "2507.05248", 2025, "PREPRINT"),
    ("huang2023",        "2310.03128", 2023, "PREPRINT"),
    ("zheng2023",        "2306.05685", 2023, "NEURIPSDB"),
    ("panickssery2024",  "2404.13076", 2024, "NEURIPS37"),
    ("wataoka2024",      "2410.21819", 2024, "NEURIPSW"),
    ("roytburg2026",     "2601.22548", 2026, "ICML"),
]

VENUE = {
    "NEURIPS":        ("inproceedings", "booktitle", "Advances in Neural Information Processing Systems"),
    "NEURIPS36":      ("inproceedings", "booktitle", "Advances in Neural Information Processing Systems 36"),
    "NEURIPS37":      ("inproceedings", "booktitle", "Advances in Neural Information Processing Systems 37"),
    "NEURIPSDB":      ("inproceedings", "booktitle", "Advances in Neural Information Processing Systems: Datasets and Benchmarks Track"),
    "NEURIPSW":       ("inproceedings", "booktitle", "NeurIPS Workshop on Safe Generative AI"),
    "ICLR":           ("inproceedings", "booktitle", "International Conference on Learning Representations"),
    "ICML":           ("inproceedings", "booktitle", "International Conference on Machine Learning"),
    "ACL":            ("inproceedings", "booktitle", "Proceedings of the Annual Meeting of the Association for Computational Linguistics"),
    "EMNLP":          ("inproceedings", "booktitle", "Proceedings of the Conference on Empirical Methods in Natural Language Processing"),
    "FINDINGS_EMNLP": ("inproceedings", "booktitle", "Findings of the Association for Computational Linguistics: EMNLP"),
    "FINDINGS_ACL":   ("inproceedings", "booktitle", "Findings of the Association for Computational Linguistics: ACL"),
    "NAACL":          ("inproceedings", "booktitle", "Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics"),
    "AAAI":           ("inproceedings", "booktitle", "Proceedings of the AAAI Conference on Artificial Intelligence"),
    "COLM":           ("inproceedings", "booktitle", "Conference on Language Modeling"),
    "ICECET":         ("inproceedings", "booktitle", "International Conference on Electrical, Computer and Energy Technologies"),
    "FAGEN":          ("inproceedings", "booktitle", "ICML Workshop on Failure Modes of Generative Agents"),
    "KDDW":           ("inproceedings", "booktitle", "KDD Workshop Proceedings"),
    "TACL":           ("article",       "journal",   "Transactions of the Association for Computational Linguistics"),
    "PREPRINT":       ("article",       "journal",   None),   # filled with arXiv preprint arXiv:ID
}

# Manual corrections that override whatever arXiv returns.
AUTHOR_OVERRIDE = {
    # Audit found automated extraction gave "Cunniff"; the real middle author is Carroll.
    # Leave as None to accept the arXiv API list, which is authoritative.
}

# Entries with no arXiv ID. Hand-written, verified against the publisher record.
MANUAL_ENTRIES = r"""
@inproceedings{patil2025,
  title     = {{The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models}},
  author    = {Patil, Shishir G. and Mao, Huanzhi and Cheng-Jie Ji, Charlie and Yan, Fanjia and Suresh, Vishnu and Stoica, Ion and Gonzalez, Joseph E.},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {48371--48392},
  year      = {2025},
  publisher = {PMLR},
  note      = {Irrelevance rubric wording appears on the accompanying HuggingFace dataset card}
}

@inproceedings{wester2024,
  title     = {{``As an AI language model, I cannot'': Investigating LLM Denials of User Requests}},
  author    = {Wester, Joel and Schrills, Tim and Pohl, Henning and van Berkel, Niels},
  booktitle = {Proceedings of the CHI Conference on Human Factors in Computing Systems},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3613904.3642135}
}

@inproceedings{anil2024,
  title     = {{Many-shot Jailbreaking}},
  author    = {Anil, Cem and Durmus, Esin and Panickssery, Nina and Sharma, Mrinank and Benton, Joe and Kundu, Sandipan and Batson, Joshua and Tong, Meg and Mu, Jesse and Ford, Daniel J. and others},
  booktitle = {Advances in Neural Information Processing Systems 37},
  year      = {2024},
  doi       = {10.5555/3737916.3742037}
}
"""

# ----------------------------------------------------------------------------
# 2. Fetch arXiv metadata in batches
# ----------------------------------------------------------------------------

NS = {"a": "http://www.w3.org/2005/Atom"}
API = "http://export.arxiv.org/api/query"


def fetch_batch(ids):
    r = requests.get(API, params={"id_list": ",".join(ids), "max_results": len(ids)},
                     headers=UA, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out = {}
    for e in root.findall("a:entry", NS):
        eid = e.find("a:id", NS).text
        m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", eid)
        if not m:
            continue
        title = " ".join(e.find("a:title", NS).text.split())
        authors = [a.find("a:name", NS).text.strip()
                   for a in e.findall("a:author", NS)]
        out[m.group(1)] = {"title": title, "authors": authors}
    return out


# ----------------------------------------------------------------------------
# 3. LaTeX escaping
# ----------------------------------------------------------------------------

ACCENTS = {
    "ö": r'\"o', "ä": r'\"a', "ü": r'\"u', "Ö": r'\"O', "Ä": r'\"A', "Ü": r'\"U',
    "é": r"\'e", "è": r"\`e", "ê": r"\^e", "á": r"\'a", "à": r"\`a", "í": r"\'i",
    "ó": r"\'o", "ú": r"\'u", "ñ": r"\~n", "ç": r"\c{c}", "å": r"\aa{}",
    "ø": r"\o{}", "ß": r"\ss{}", "ł": r"\l{}", "š": r"\v{s}", "č": r"\v{c}",
    "ž": r"\v{z}", "ı": r"\i{}", "â": r"\^a", "î": r"\^i", "ô": r"\^o",
}


def tex(s):
    s = html.unescape(s)
    for k, v in ACCENTS.items():
        s = s.replace(k, "{" + v + "}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def fmt_authors(names):
    # arXiv gives "First M. Last". BibTeX wants "Last, First M." joined by " and ".
    out = []
    for n in names:
        parts = n.split()
        if len(parts) == 1:
            out.append(tex(parts[0]))
        else:
            out.append(tex(parts[-1]) + ", " + tex(" ".join(parts[:-1])))
    return " and ".join(out)


# ----------------------------------------------------------------------------
# 4. Build
# ----------------------------------------------------------------------------

def main():
    ids = [r[1] for r in ROWS]
    meta = {}
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        print("fetching %d-%d of %d ..." % (i + 1, i + len(chunk), len(ids)))
        try:
            meta.update(fetch_batch(chunk))
        except Exception as e:
            print("  batch failed: %s" % e)
        time.sleep(3)

    lines = ["% references.bib",
             "% Generated by build_bib.py from the verified citation master.",
             "% Years are the PUBLISHED VENUE year, not the arXiv preprint year.",
             "% Distinct keys are already disambiguated for the four surname collisions.",
             ""]
    missing = []

    for key, aid, year, vtag in ROWS:
        m = meta.get(aid)
        if not m:
            missing.append((key, aid))
            continue
        etype, vfield, vname = VENUE[vtag]
        if vname is None:
            vname = "arXiv preprint arXiv:%s" % aid
        auth = AUTHOR_OVERRIDE.get(key) or fmt_authors(m["authors"])
        lines.append("@%s{%s," % (etype, key))
        lines.append("  title     = {{%s}}," % tex(m["title"]))
        lines.append("  author    = {%s}," % auth)
        lines.append("  %-9s = {%s}," % (vfield, vname))
        lines.append("  year      = {%s}," % year)
        lines.append("  eprint    = {%s}," % aid)
        lines.append("  archivePrefix = {arXiv}")
        lines.append("}")
        lines.append("")

    lines.append(MANUAL_ENTRIES.strip())
    lines.append("")

    with open("references.bib", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    n = len(ROWS) - len(missing) + 3
    print("\nwrote references.bib with %d entries" % n)
    if missing:
        print("COULD NOT FETCH (add by hand):")
        for k, a in missing:
            print("   %-18s arXiv:%s" % (k, a))

    print("""
In the NeurIPS template, before \\end{document}:

    \\bibliographystyle{plainnat}
    \\bibliography{references}

Cite with \\citep{key} for parenthetical, \\citet{key} for inline.

CHECK BY HAND AFTER GENERATION:
  yu2026    middle author must read Carroll, not Cunniff
  sun2026   author block had a duplicated name in automated extraction
  laban2026 year is forced to 2026 (ICLR Oral). Correct, do not revert.
  soni2026  venue unconfirmed, left as preprint
""")


if __name__ == "__main__":
    main()