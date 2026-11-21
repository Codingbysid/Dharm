"""Decision Corpus v1.0. Run --help or --check_data without downloading models."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile


BI = "BAAI/bge-small-en-v1.5"
CE = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path, key):
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), f"Duplicate {key} in {path}")
    return dict(sorted(result.items()))


def load_training(folder):
    records = read_rows(folder / "decisions_v0.2.jsonl", "id")
    require(set(records) == {f"D{i:03}" for i in range(1, 109)}, "Expected D001–D108")
    for k, r in records.items():
        require(len(r) == 24 and len(r["scenarios_json"]) == 2, f"Unexpected schema for {k}")
        for edge in (CONTRAST, WORKFLOW):
            require(r[edge]["id"] in records and r[edge]["id"] != k, f"Broken {edge} at {k}")
        peer = records[r[WORKFLOW]["id"]][WORKFLOW]
        role = r[WORKFLOW]["sequence_role"]
        require(peer["id"] == k and
                ((role.startswith("Follow with") and peer["sequence_role"].startswith("Precede with"))
                 or (role.startswith("Precede with") and peer["sequence_role"].startswith("Follow with"))),
                f"Inconsistent workflow at {k}")
    notes = read_rows(folder / "chanakya_daily_supplement.jsonl", "note_id")
    require(set(notes) == {f"CD{i:02}" for i in range(1, 10)}, "Expected CD01–CD09")
    for note, row in notes.items():
        target = row["existing_record_id"]
        require(target == SUPPLEMENT.get(note), f"Unexpected mapping for {note}")
        if target:
            records[target]["scenarios_json"].append({"scenario": row["modern_application"]})
    return records


def passage_card(r):
    return (f"[{r['id']}] {r['work']} {r['locator']} | Domain: {r['decision_domain']} | "
            f"Urgency: {r['urgency_level']} | Reversibility: {r['reversibility']}\n"
            f"Principle: {r['derived_principle']}\nContext: {r['context_summary']}\n"
            f"Text: {r['english_translation']}\nCues: {r['retrieval_cues']}")


def arbitrator_card(r):
    edge = r[CONTRAST]
    return (f"Principle: {r['derived_principle']}\nRetain: {edge['retain_self_when']}\n"
            f"Flip: {edge['flip_to_competing_when']}\nLimit: {r['application_limit']}\n"
            f"Workflow: {r[WORKFLOW]['sequence_role']}")


def anchors(records):
    return [(s["scenario"], k) for k, r in records.items() for s in r["scenarios_json"]]


def empirical():
    """Six families with ten distinct queries each, independent of evaluation files."""
    out = []
    instruments = ["balance", "voltmeter", "barometer", "photometer", "tachometer",
                   "manometer", "flowmeter", "encoder", "load cell", "spectrometer"]
    materials = ["wax", "graphite", "mica", "slate", "linen", "pine", "opal", "rubber", "felt", "quartz"]
    for i, (instrument, material) in enumerate(zip(instruments, materials), 1):
        out.extend([f"What is the product of {11+i} and {37+i}?",
                    f"Give the recorded year in which manuscript series {800+i} was first catalogued.",
                    f"Which workshop manufactured accession {1700+i} in the archive?",
                    f"What was the indicated offset of the {instrument} against its reference standard?",
                    f"What density did the analyst measure for specimen {i} of {material}?",
                    f"How many seconds elapsed across {30+i} intervals of {2*i} seconds each?"])
    return out


def split_records(records, dev):
    if not dev:
        return records, {}
    # Whole closed six-record blocks. Every contrast and workflow endpoint stays in its split.
    held = set()
    for start in (43, 67, 85, 103):
        held.update(f"D{i:03}" for i in range(start, start+6))
    for k, r in records.items():
        for edge in (CONTRAST, WORKFLOW):
            require((k in held) == (r[edge]["id"] in held), "Development split crosses an edge")
    return ({k:r for k,r in records.items() if k not in held},
            {k:r for k,r in records.items() if k in held})


def expand(seed, records):
    return sorted({j for k in seed for j in (k, records[k][CONTRAST]["id"], records[k][WORKFLOW]["id"])})


def arbitrate(pool, rel, abst, records):
    raw = dict(zip(pool, rel, strict=True))
    adjusted = raw.copy()
    for k in pool:
        edge = records[k][WORKFLOW]
        j = edge["id"]
        if edge["sequence_role"].startswith("Follow with") and min(raw[k], raw.get(j, 0)) > .50:
            adjusted[k] += .03
            adjusted[j] += .015
    top = sorted(pool, key=lambda k: (-adjusted[k], k))[:3]
    decline = max(abst) > .55 and max(rel) < .45
    return dict(primary="Abstain" if decline else top[0], top3=top, abstained=decline,
                pool=pool, relevance=raw, abstention=dict(zip(pool, abst)), adjusted=adjusted)


def pair_examples(train):
    # Accumulate by query/card so reciprocal triggers or repeated scenarios cannot overweight copies.
    examples = {}
    def add(q, k, relevance, abstain):
        prev = examples.get((q,k))
        if prev:
            require(prev[0] == relevance, f"Contradictory relevance labels at {k}")
            if prev[1] is not None:
                require(abstain in (None,prev[1]), "Conflicting scope labels")
                abstain = prev[1]
        examples[q,k] = (relevance,abstain)
    for k,r in train.items():
        edge = r[CONTRAST]
        j = edge["id"]
        for s in r["scenarios_json"]:
            add(s["scenario"],k,1,0)
            add(s["scenario"],j,0,0)
        add(r["misapplication_scenario"],k,0,None)
        for field, label in (("retain_self_when",1),("flip_to_competing_when",0)):
            add(edge[field],k,label,None)
            add(edge[field],j,1-label,None)
    return [(q,k,y,a) for (q,k),(y,a) in examples.items()]


def batches(items,n):
    return [items[i:i+n] for i in range(0,len(items),n)]


def schedule(optimizer, steps):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(optimizer, math.ceil(.1*steps), steps)


def device_for(choice):
    import torch
    available = {"cpu":True,"cuda":torch.cuda.is_available(),"mps":torch.backends.mps.is_available()}
    if choice == "auto":
        return next(d for d in ("cuda","mps","cpu") if available[d])
    require(available[choice], f"Requested device unavailable: {choice}")
    return choice


def train_bi(train, args, device):
    import torch
    from torch.nn import functional as F
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    model = SentenceTransformer(BI, device=device)
    model.max_seq_length = args.max_length
    data = anchors(train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = schedule(optimizer, args.epochs_bi*math.ceil(len(data)/args.batch_size))
    def encode(texts):
        tokens = {k:v.to(device) for k,v in model.tokenize(texts).items()}
        return F.normalize(model(tokens)["sentence_embedding"],dim=-1)
    cards = {k:passage_card(r) for k,r in train.items()}
    for epoch in range(args.epochs_bi):
        model.train()
        random.Random(args.seed+epoch).shuffle(data)
        for batch in tqdm(batches(data,args.batch_size), desc=f"Retriever {epoch+1}"):
            queries, keys = zip(*batch)
            # Unique positive columns prevent same-passage scenarios becoming false negatives.
            unique = list(dict.fromkeys(keys))
            labels = torch.tensor([unique.index(k) for k in keys],device=device)
            q = encode([QUERY_PREFIX+t for t in queries])
            p = encode([cards[k] for k in unique])
            positive = p[labels]
            boundary = encode([train[k]["misapplication_scenario"] for k in keys])
            rival = encode([cards[train[k][CONTRAST]["id"]] for k in keys])
            # Multiple-negatives ranking loss (cosine, scale 20) plus two cosine triplets.
            ranking = F.cross_entropy(20*(q @ p.T),labels)
            sim_pos = (q*positive).sum(-1)
            triplet = sum(F.relu(.25+(q*n).sum(-1)-sim_pos).mean() for n in (boundary,rival))/2
            loss = ranking + args.triplet_weight*triplet
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            scheduler.step()
    return model.eval()


class DenseIndex:
    def __init__(self, model, records, batch_size):
        self.model = model
        self.keys = sorted(records)
        self.batch_size = batch_size
        self.matrix = self.encode([passage_card(records[k]) for k in self.keys])
    def encode(self,texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
    def search(self, queries, allowed=None):
        scores = self.encode([QUERY_PREFIX+t for t in queries]) @ self.matrix.T
        candidates = [i for i,k in enumerate(self.keys) if allowed is None or k in allowed]
        return [[self.keys[i] for i in sorted(candidates,key=lambda i:(-row[i],self.keys[i]))[:5]]
                for row in scores]


def train_ce(train, index, args, device):
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained(CE)
    model = AutoModelForSequenceClassification.from_pretrained(
        CE,num_labels=2,ignore_mismatched_sizes=True,id2label={0:"relevance",1:"abstain"},
        label2id={"relevance":0,"abstain":1}).to(device)
    examples = pair_examples(train)
    outside = empirical()
    if args.dev:
        outside = [q for i,q in enumerate(outside) if i%6 != 2]
    for q,neighbors in zip(outside,index.search(outside,train)):
        examples.extend((q,k,0,1) for k in neighbors[:3])
    cards = {k:arbitrator_card(r) for k,r in train.items()}
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr)
    sched = schedule(opt, args.epochs_ce*math.ceil(len(examples)/args.batch_size))
    model.train()
    for epoch in range(args.epochs_ce):
        random.Random(args.seed+epoch).shuffle(examples)
        for batch in tqdm(batches(examples,args.batch_size),desc=f"Arbitrator {epoch+1}"):
            q,k,rel,ab = zip(*batch)
            tokens = tokenizer(list(q),[cards[j] for j in k],padding=True,truncation=True,
                               max_length=args.max_length,return_tensors="pt").to(device)
            logits = model(**tokens).logits
            def targets(values):
                return torch.tensor(values,dtype=logits.dtype,device=device)
            relevance_loss = F.binary_cross_entropy_with_logits(logits[:,0], targets(rel))
            # Unspecified scope is unknown, especially for a valid dilemma paired with a wrong principle.
            mask = targets([float(v is not None) for v in ab])
            scope_loss = F.binary_cross_entropy_with_logits(logits[:,1],targets([v or 0 for v in ab]),reduction="none")
            loss = relevance_loss + args.scope_weight*(scope_loss*mask).sum()/mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            sched.step()
    return model.eval(),tokenizer


def predict(questions, records, index, model, tokenizer, args, device):
    import torch
    texts = [q["dilemma_text"] for q in questions]
    results = []
    with torch.inference_mode():
        for q,text,seed in zip(questions,texts,index.search(texts)):
            pool = expand(seed,records)
            scores = []
            for block in batches(pool,args.batch_size):
                tokens = tokenizer([text]*len(block),[arbitrator_card(records[k]) for k in block],
                                   padding=True,truncation=True,max_length=args.max_length,return_tensors="pt").to(device)
                scores.extend(model(**tokens).logits.sigmoid().cpu().tolist())
            rel,ab = zip(*scores)
            results.append(dict(q, seeds=seed, **arbitrate(pool,rel,ab,records)))
    return results


def evaluate(rows):
    recall,ordered,pairs = [],[],defaultdict(list)
    tp=pred=gold=0
    for r in rows:
        p,c=r["gold_primary_id"],r["gold_competing_id"]
        pred+=r["abstained"]
        if r["query_type"]=="abstention":
            gold+=1
            tp+=r["abstained"]
            continue
        relevant={p}
        if r["query_type"]=="synthesis":
            relevant.add(c)
            ordered.append(not r["abstained"] and r["top3"][:2]==[p,c])
        elif r.get("pair_group"):
            pairs[r["pair_group"]].append(r["primary"]==p)
        recall.append(len(relevant.intersection(r["top3"]))/len(relevant))
    require(all(len(group)==2 for group in pairs.values()),"Incomplete arbitration pair")
    return dict(recall_at_3=sum(recall)/len(recall),flip=[sum(all(g) for g in pairs.values()),len(pairs)],
                abstention_precision=[tp,pred],abstention_recall=[tp,gold],ordered_synthesis=[sum(ordered),len(ordered)])


def benchmark(folder,records):
    qs=read_rows(folder/"eval_queries_v0.2.jsonl","query_id")
    require(list(qs)==[f"Q{i:02}" for i in range(1,43)],"Expected Q01–Q42")
    for q in qs.values():
        if q["query_type"]=="abstention":
            require(q["gold_primary_id"] is None and q["gold_competing_id"] is None,"Invalid abstention")
        else:
            require(all(q[k] in records for k in ("gold_primary_id","gold_competing_id")),"Unresolved gold ID")
    return list(qs.values())


def dev_questions(dev):
    qs=[]
    for text,k in anchors(dev):
        qs.append(dict(query_type="scenario",dilemma_text=text,gold_primary_id=k,gold_competing_id=None))
    for i,text in enumerate(empirical()):
        if i%6==2:
            qs.append(dict(query_type="abstention",dilemma_text=text,gold_primary_id=None,gold_competing_id=None))
    return [dict(q,query_id=f"dev-{i}") for i,q in enumerate(qs,1)]


def show(name,model,m):
    def fraction(k):
        a,b=m[k]
        return f"{a}/{b}" if b else "undefined"
    print(f"| {name} | {model} | {100*m['recall_at_3']:.2f}% | "+
          " | ".join(fraction(k) for k in ("flip","abstention_precision","abstention_recall","ordered_synthesis"))+" |")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs_bi",type=int,default=6)
    ap.add_argument("--epochs_ce",type=int,default=5)
    ap.add_argument("--device",choices=["auto","cuda","mps","cpu"],default="auto")
    ap.add_argument("--data_dir",type=Path,default=Path("data"))
    ap.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    ap.add_argument("--batch_size",type=int,default=16)
    ap.add_argument("--max_length",type=int,default=384)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--triplet_weight",type=float,default=1.)
    ap.add_argument("--scope_weight",type=float,default=1.)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--check_data",action="store_true")
    ap.add_argument("--dev",action="store_true",help="Withhold four closed blocks and an empirical template family. Never open the benchmark.")
    args=ap.parse_args()
    require(min(args.epochs_bi,args.epochs_ce,args.batch_size)>0 and 32<=args.max_length<=512 and
            args.lr>0 and min(args.triplet_weight,args.scope_weight)>=0,"Invalid training settings")
    records=load_training(args.data_dir)
    train,dev=split_records(records,args.dev)
    pairs=pair_examples(train)
    print(f"{len(records)} passages, {len(anchors(train))} training anchors, {len(pairs)} labeled relevance pairs, {len(dev)} dev passages.")
    require(len(anchors(records))==222 and len(set(empirical()))==60,"Unexpected example count")
    if args.check_data:
        return
    import numpy as np
    import torch
    device=device_for(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Training on {device}",flush=True)
    dest=args.artifacts.resolve()
    require(not dest.exists() or (dest.is_dir() and not any(dest.iterdir())),f"Choose an empty --artifacts directory: {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".decision-run-",dir=dest.parent) as tmp:
        stage=Path(tmp)/"release"
        stage.mkdir()
        bi=train_bi(train,args,device)
        bi.save(str(stage/"bi_encoder"))
        index=DenseIndex(bi,records,args.batch_size)
        # Compute the index once. With --dev, mining is restricted to training endpoints.
        model,tok=train_ce(train,index,args,device)
        model.save_pretrained(stage/"cross_encoder_arbitrator")
        tok.save_pretrained(stage/"cross_encoder_arbitrator")
        questions=dev_questions(dev) if args.dev else benchmark(args.data_dir,records)
        rows=predict(questions,records,index,model,tok,args,device)
        slices={"dev":rows} if args.dev else {"original":rows[:18],"new":rows[18:],"all":rows}
        baseline={}
        if not args.dev:
            for name,vals in zip(slices,[(.8125,[4,6],[2,2],[2,2],[2,4]),
                                         (14.5/22,[0,8],[0,0],[0,2],[3,6]),
                                         (27.5/38,[4,14],[2,2],[2,4],[5,10])]):
                baseline[name]=dict(zip(["recall_at_3","flip","abstention_precision","abstention_recall","ordered_synthesis"],vals))
        print("| Slice | Model | Recall@3 | Correct flips | Abstention precision | Abstention recall | Ordered synthesis |")
        print("|---|---|---:|---:|---:|---:|---:|")
        metrics={k:evaluate(v) for k,v in slices.items()}
        for name,m in metrics.items():
            if baseline:show(name,"TF-IDF v0.2.3",baseline[name])
            show(name,"Dense + arbitrator",m)
        inputs=["decisions_v0.2.jsonl","chanakya_daily_supplement.jsonl"]
        if not args.dev:inputs.append("eval_queries_v0.2.jsonl")
        report=dict(config=vars(args),device=device,models=[BI,CE],torch_version=torch.__version__,
                    fingerprints={n:hashlib.sha256((args.data_dir/n).read_bytes()).hexdigest() for n in inputs},
                    baseline=baseline,metrics=metrics,predictions=rows)
        name="dev_report.json" if args.dev else "v1_ml_eval_report.json"
        (stage/name).write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str,allow_nan=False)+"\n",encoding="utf8")
        stage.replace(dest)
    print(f"Saved models and {name} to {dest}")


if __name__=="__main__":
    main()
