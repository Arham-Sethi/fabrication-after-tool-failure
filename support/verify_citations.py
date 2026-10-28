#!/usr/bin/env python3
"""Independent citation checker for both papers.

Reads citations_master.csv, fetches each arXiv abstract page, and prints the
TRUE title, authors and date next to what you are citing it as, so you can
eyeball every row yourself rather than trusting anyone's summary.

It flags three things automatically:
  1. an ID that does not resolve at all
  2. a year mismatch between your citation and the arXiv submission date
  3. a first-author surname that does not appear in the fetched author list

It cannot check whether the paper SUPPORTS your claim. That is the
what_we_cite_it_for column, and only you can verify it by reading. The script
prints that column beside each result so you can check as you go.

USAGE
    pip install requests
    python verify_citations.py                 # check everything
    python verify_citations.py --paper P1      # one paper
    python verify_citations.py --only 2404.13208
    python verify_citations.py --out report.csv

Rate limited to 1 request per 3 seconds, which arXiv asks for. A full run of
55 rows takes about 2 minutes.
"""
from __future__ import annotations
import argparse, csv, html, io, os, re, sys, time

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

HERE = os.path.dirname(os.path.abspath(__file__))

# The citation table is embedded so there is no second file to download.
CSV_DATA = r"""paper,key,cited_as,what_we_cite_it_for,id_type,id,venue_as_verified,status,YOU_MUST_CHECK
P1,schick2023,"Schick et al., 2023","Toolformer; self-supervised API invocation",arxiv,2302.04761,"NeurIPS 36",VERIFIED,
P1,qin2024,"Qin et al., 2024","ToolLLM; 16000 real-world APIs",arxiv,2307.16789,"ICLR 2024",VERIFIED,
P1,li2023,"Li et al., 2023","API-Bank; 73 APIs, 314 dialogues",arxiv,2304.08244,"EMNLP 2023",VERIFIED-PARTIAL,"full author list"
P1,patil2025,"Patil et al., 2025","BFCL; function-calling standard + irrelevance rubric",pmlr,"v267 pp.48371-48392","ICML 2025",VERIFIED,"NO arXiv ID. Cite PMLR. Irrelevance definition is on the HF dataset card, NOT in the paper. Cite both."
P1,mialon2023,"Mialon et al., 2023","GAIA; humans 92% vs GPT-4+plugins 15%",arxiv,2311.12983,"ICLR 2024",VERIFIED,
P1,liuX2024,"Liu et al., 2024","AgentBench; 8 environments",arxiv,2308.03688,"ICLR 2024",VERIFIED,"NAME COLLISION with Liu et al. 2026. Bibliography must disambiguate."
P1,zhouS2024,"Zhou et al., 2024","WebArena; 14.41% vs 78.24%; unachievable tasks",arxiv,2307.13854,"ICLR 2024",VERIFIED,"NAME COLLISION with Zhou et al. 2023 (different Zhou). Disambiguate."
P1,yao2024,"Yao et al., 2024","tau-bench; policy-constrained domains",arxiv,2406.12045,"preprint",VERIFIED-PARTIAL,"full author list"
P1,hossain2026,"Hossain et al., 2026","only prior framework audit; architectural containment, 3 frameworks",arxiv,2606.12797,"preprint",VERIFIED,
P1,chenJ2024,"Chen et al., 2024","RGB; negative rejection; models struggle significantly",arxiv,2309.01431,"AAAI 2024",VERIFIED,"NAME COLLISION with Chen et al. 2025 (different Chen). Disambiguate."
P1,ming2025,"Ming et al., 2025","FaithEval; unanswerable/inconsistent/counterfactual contexts",arxiv,2410.03727,"ICLR 2025",VERIFIED-PARTIAL,"author ordering"
P1,wu2024,"Wu et al., 2024","ClashEval; overrides correct prior knowledge >60%",arxiv,2404.10198,"NeurIPS D&B",VERIFIED-PARTIAL,"full author list"
P1,wang2025,"Wang et al., 2025","Astute RAG; RAG can underperform no-RAG under imperfect retrieval",arxiv,2410.07176,"ACL 2025",VERIFIED,"REWORD: claim is made via worst-case comparison, not stated directly"
P1,yoran2024,"Yoran et al., 2024","training-time robustness to irrelevant context",arxiv,2310.01558,"ICLR 2024",VERIFIED,
P1,zhouW2023,"Zhou et al., 2023","Context-faithful prompting; prompt-time mitigation",arxiv,2303.11315,"Findings EMNLP 2023",VERIFIED,"NAME COLLISION with Zhou et al. 2024. Different first author. Disambiguate."
P1,sharma2023,"Sharma et al., 2023","sycophancy; preference judgments",arxiv,2310.13548,"ICLR 2024",VERIFIED,"HEDGE REQUIRED: paper says 'likely driven in part by'. Do not write 'linked to' as causal."
P1,rottger2024,"Röttger et al., 2024","XSTest; 250 safe + 200 unsafe; safety training raises benign refusal",arxiv,2308.01263,"NAACL 2024",VERIFIED,
P1,sclar2024,"Sclar et al., 2024","prompt formatting; spreads up to 76 accuracy points",arxiv,2310.11324,"ICLR 2024",VERIFIED,
P1,wallace2024,"Wallace et al., 2024","instruction hierarchy; tool output = lowest privilege",arxiv,2404.13208,"preprint (OpenAI)",VERIFIED,"Table 3: 'Priority 30 (low): Text from tools'"
P1,eliav2026,"Eliav, 2026","instruction load vs content; fabrication 0/5760",arxiv,2607.19257,"preprint",VERIFIED,
P1,kadavath2022,"Kadavath et al., 2022","partial self-knowledge; P(True)",arxiv,2207.05221,"preprint (Anthropic)",VERIFIED,"36 authors"
P1,yin2023,"Yin et al., 2023","detects unanswerable questions; sub-human",arxiv,2305.18153,"Findings ACL 2023",VERIFIED,"TITLE: 'Do LLMs Know What They Don't Know?'. SelfAware is the DATASET."
P1,kirichenko2025,"Kirichenko et al., 2025","AbstentionBench; reasoning FT degrades abstention 24%",arxiv,2506.09038,"preprint (FAIR)",VERIFIED,
P1,wen2024,"Wen et al., 2024","abstention survey; query/model/human-values axes",arxiv,2407.18418,"TACL",VERIFIED-PARTIAL,"confirm TACL venue"
P1,turpin2023,"Turpin et al., 2023","unfaithful CoT; omits biasing features",arxiv,2305.04388,"NeurIPS 2023",VERIFIED,
P1,lanham2023,"Lanham et al., 2023","CoT faithfulness varies; decreases with scale",arxiv,2307.13702,"preprint (Anthropic)",VERIFIED,
P1,chenY2025,"Chen et al., 2025","reasoning models; reveal rate often below 20%",arxiv,2505.05410,"preprint (Anthropic)",VERIFIED,"REWORD: 'often below 20%', not 'fewer than 20%'"
P1,rao2026,"Rao and Callison-Burch, 2026","judge agreement metrics; Cohen kappa adds information",arxiv,2606.00093,"preprint",VERIFIED,"USE THIS for the kappa argument, NOT 2406.12624"
P1,advani2026,"Advani, 2026","no judge config exceeds AUROC 0.65 on false success",arxiv,2606.09863,"FAGEN@ICML 2026",VERIFIED,"0.83 on tau2-bench, 0.95 on AppWorld for the TF-IDF baseline"
P1,zhu2026,"Zhu et al., 2026","tool perturbation; recovery metrics; 'scaling or prompting'",arxiv,2606.05806,"preprint",VERIFIED,"TITLE is 'When Tools Fail: ...'. ToolMaze is the BENCHMARK name."
P1,soni2026,"Soni, 2026","ToolFailBench; parametric traps; fixed prompt",arxiv,2607.04686,"preprint",VERIFIED-PARTIAL,"VENUE UNCONFIRMED. Renders in ICML template. Cite as preprint until confirmed. Prompt is fixed PER DOMAIN across 5 domains."
P1,liu2026abstain,"Liu et al., 2026","AgentAbstain; abstention from acting; 42 sandboxes",arxiv,2607.10059,"preprint",VERIFIED,"NAME COLLISION with Liu et al. 2024 and with Liu et al. 2026 in Paper 2. Disambiguate."
P1,singh2026,"Singh, 2026","silent failures; 56.6% fabrication; 15.6x USR; keyword detector",arxiv,2607.19449,"KDD 2026 workshop",VERIFIED,"READ IN FULL before submission"
P1,tamba2026,"Tamba, 2026","field evidence; exit 143 recorded as confirmed results",arxiv,2607.13071,"preprint",VERIFIED,"NOT in your final text. Recommended for the Introduction."
P2,rottger2024,"Röttger et al., 2024","XSTest; design template for a contrast set",arxiv,2308.01263,"NAACL 2024",VERIFIED,
P2,cui2025,"Cui et al., 2025","OR-Bench; 80000 prompts, 32 models",arxiv,2405.20947,"ICML 2025",VERIFIED,
P2,zhang2025,"Zhang et al., 2025","FalseReject; over-refusal is a policy artifact",arxiv,2505.08054,"COLM 2025",VERIFIED,"NAME COLLISION with Zhang et al. 2026 if you add the tool-use tax paper."
P2,vanderweij2024,"van der Weij et al., 2024","sandbagging; requires instruction or fine-tune",arxiv,2406.07358,"preprint",VERIFIED,
P2,wester2024,"Wester et al., 2024","technical inability vs policy restriction denials",doi,10.1145/3613904.3642135,"CHI 2024",VERIFIED,"NO arXiv. Cite the ACM DOI."
P2,arditi2024,"Arditi et al., 2024","refusal mediated by a single direction; 13 models",arxiv,2406.11717,"NeurIPS 2024",VERIFIED,
P2,yu2026,"Yu et al., 2026","tool affordance; violations up to 85% under identical policies",arxiv,2603.20320,"ICECET 2026",VERIFIED,"AUTHOR FIX: Yu, CARROLL, Bentley. Not 'Cunniff'. TITLE ends 'in LLM Agents'."
P2,cartagena2026,"Cartagena and Teixeira, 2026","text and tool-call channels dissociate; 17420 datapoints",arxiv,2602.16943,"preprint",VERIFIED,
P2,zeng2026,"Zeng et al., 2026","tools degrade internally-solvable accuracy 3.29-14.48%",arxiv,2604.19749,"preprint",VERIFIED-PARTIAL,"abs page returned no machine-readable text; confirmed via HTML full text"
P2,cheng2026,"Cheng et al., 2026","tool-necessity mismatch 26.5-54.0%; knowing-doing gap",arxiv,2605.14038,"preprint",VERIFIED-PARTIAL,"confirmed via HTML full text"
P2,sun2026,"Sun et al., 2026","probes predict tool necessity AUROC 0.89-0.96",arxiv,2605.09252,"preprint",VERIFIED-PARTIAL,"RE-CHECK AUTHOR BLOCK. Extraction duplicated a name."
P2,liu2026irrel,"Liu et al., 2026","structural adjacency; invocation 0.16% to 41.86%",arxiv,2604.11322,"preprint",VERIFIED-PARTIAL,"abs-page date not retrievable. NAME COLLISION with Liu et al. 2026 in Paper 1."
P2,anil2024,"Anil et al., 2024","many-shot jailbreaking; in-context demos flip later refusal",doi,10.5555/3737916.3742037,"NeurIPS 37",VERIFIED,"NO arXiv ID EXISTS. Cite NeurIPS proceedings or the DOI."
P2,laban2026,"Laban et al., 2026","multi-turn drop 39%; commitment to early assumptions",arxiv,2505.06120,"ICLR 2026 Oral",VERIFIED,"CITE AS 2026, not 2025. OpenReview id VKGTGGcwl6."
P2,miao2025,"Miao et al., 2025","contextual priming; preceding response steers behaviour",arxiv,2507.05248,"preprint",VERIFIED,
P2,huang2023,"Huang et al., 2023","MetaTool; negative samples test withholding only",arxiv,2310.03128,"preprint",VERIFIED,"arXiv title differs from in-paper title"
P2,patil2025,"Patil et al., 2025","BFCL irrelevance rubric rewards declining",pmlr,"v267 pp.48371-48392","ICML 2025",VERIFIED,"Rubric wording is on the HF dataset card. Cite the card too."
P2,zheng2023,"Zheng et al., 2023","MT-Bench; ~80% agreement; position/verbosity/self-enhancement bias",arxiv,2306.05685,"NeurIPS D&B 2023",VERIFIED,
P2,panickssery2024,"Panickssery et al., 2024","judges recognize and favor own generations; 73.5% self-recognition",arxiv,2404.13076,"NeurIPS 37",VERIFIED,
P2,wataoka2024,"Wataoka et al., 2024","self-preference driven by low-perplexity preference",arxiv,2410.21819,"NeurIPS 2024 Safe GenAI Workshop",VERIFIED,
P2,roytburg2026,"Roytburg et al., 2026","controlling evaluator quality; ~51% of prior findings survive",arxiv,2601.22548,"ICML 2026",VERIFIED,"YOUR TEXT SAYS 'substantially reduces these effects'. The paper says half the findings LOSE SIGNIFICANCE. Reword."
"""
UA = {"User-Agent": "citation-check/1.0 (academic verification; single user)"}
DELAY = 3.0

RE_TITLE = re.compile(r'<meta name="citation_title" content="([^"]*)"')
RE_AUTHOR = re.compile(r'<meta name="citation_author" content="([^"]*)"')
RE_DATE = re.compile(r'<meta name="citation_date" content="([^"]*)"')
RE_H1 = re.compile(r'<h1 class="title[^"]*">(?:<span[^>]*>[^<]*</span>)?(.*?)</h1>', re.S)


def strip(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def fetch_arxiv(arxiv_id: str):
    """Return dict with title, authors, date, or an error string."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        r = requests.get(url, headers=UA, timeout=30)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if r.status_code == 404:
        return {"error": "404 DOES NOT EXIST"}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    t = r.text
    title = strip(RE_TITLE.search(t).group(1)) if RE_TITLE.search(t) else None
    if not title:
        m = RE_H1.search(t)
        title = strip(m.group(1)) if m else None
    authors = [strip(a) for a in RE_AUTHOR.findall(t)]
    date = RE_DATE.search(t).group(1) if RE_DATE.search(t) else None
    if not title:
        return {"error": "fetched but could not parse title"}
    return {"title": title, "authors": authors, "date": date, "url": url}


def surname_of(cited_as: str) -> str:
    """'van der Weij et al., 2024' -> 'weij'; 'Rao and Callison-Burch, 2026' -> 'rao'."""
    head = cited_as.split(",")[0]
    head = re.sub(r"\bet al\.?", "", head)
    head = re.split(r"\band\b", head)[0]
    parts = [p for p in head.strip().split() if p]
    return parts[-1].lower() if parts else ""


def year_of(cited_as: str):
    m = re.search(r"(19|20)\d{2}", cited_as)
    return int(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=["P1", "P2"])
    ap.add_argument("--only")
    ap.add_argument("--out", default=os.path.join(HERE, "verification_report.csv"))
    a = ap.parse_args()

    rows = list(csv.DictReader(io.StringIO(CSV_DATA.strip("\n"))))
    if not rows:
        sys.exit("embedded table failed to parse; re-download this file")
    print(f"loaded {len(rows)} citations from the embedded table")
    if a.paper:
        rows = [r for r in rows if r["paper"] == a.paper]
    if a.only:
        rows = [r for r in rows if r["id"] == a.only]

    seen, out, flags = {}, [], []
    for i, r in enumerate(rows, 1):
        key, cited, idt, idv = r["key"], r["cited_as"], r["id_type"], r["id"]
        print(f"\n[{i}/{len(rows)}] {r['paper']}  {cited}")
        print(f"    citing for : {r['what_we_cite_it_for']}")

        if idt != "arxiv":
            print(f"    {idt.upper()} {idv}  -> CHECK BY HAND, not on arXiv")
            out.append({**r, "fetched_title": "", "fetched_authors": "",
                        "fetched_date": "", "auto_flag": "MANUAL"})
            continue

        if idv in seen:
            res = seen[idv]
            print("    (cached)")
        else:
            res = fetch_arxiv(idv)
            seen[idv] = res
            time.sleep(DELAY)

        if "error" in res:
            print(f"    !! {res['error']}")
            flags.append(f"{key}: {res['error']}")
            out.append({**r, "fetched_title": "", "fetched_authors": "",
                        "fetched_date": "", "auto_flag": res["error"]})
            continue

        auth = ", ".join(res["authors"][:4]) + (" ..." if len(res["authors"]) > 4 else "")
        print(f"    TITLE      : {res['title']}")
        print(f"    AUTHORS    : {auth}")
        print(f"    DATE       : {res['date']}")

        f = []
        sn = surname_of(cited)
        if sn and not any(sn in x.lower() for x in res["authors"]):
            f.append(f"FIRST AUTHOR '{sn}' NOT IN AUTHOR LIST")
        cy, dy = year_of(cited), None
        if res["date"]:
            m = re.search(r"(19|20)\d{2}", res["date"])
            dy = int(m.group(0)) if m else None
        if cy and dy and abs(cy - dy) > 0:
            f.append(f"YEAR: you cite {cy}, arXiv says {dy}")
        for x in f:
            print(f"    !! {x}")
            flags.append(f"{key}: {x}")
        if r["YOU_MUST_CHECK"]:
            print(f"    ** ACTION  : {r['YOU_MUST_CHECK']}")
        out.append({**r, "fetched_title": res["title"],
                    "fetched_authors": "; ".join(res["authors"]),
                    "fetched_date": res["date"] or "",
                    "auto_flag": " | ".join(f)})

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print("\n" + "=" * 70)
    print(f"checked {len(rows)} rows, wrote {a.out}")
    print("=" * 70)
    if flags:
        print(f"\n{len(flags)} AUTOMATIC FLAG(S):")
        for x in flags:
            print("  -", x)
    else:
        print("\nno automatic flags. IDs resolve, years and first authors match.")
    print("\nA year mismatch is often correct: arXiv shows the preprint date, you")
    print("may be citing the published venue. Laban et al. is exactly this case.")
    print("\nThe script CANNOT check whether a paper supports your claim.")
    print("Read the what_we_cite_it_for column against each abstract yourself.")


if __name__ == "__main__":
    main()