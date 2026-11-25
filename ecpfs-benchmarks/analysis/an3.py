import json,glob,os
import numpy as np
D=["ccnews","imagenet","laion","msmarco"]
Cs=[128,256,512,1024,2048,4096,8192]
def rows(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
def warm(r): return r["passes"][-1]["qps"]
def cold(r): return r["passes"][0]["qps"]
def qps_at(rs,target):
    pts=sorted([(r["recall"],warm(r)) for r in rs])
    if not pts or target<pts[0][0] or target>pts[-1][0]: return None
    for (r0,q0),(r1,q1) in zip(pts,pts[1:]):
        if r0<=target<=r1:
            t=(target-r0)/(r1-r0)
            return float(np.exp(np.log(q0)+t*(np.log(q1)-np.log(q0))))
print("=== WARM QPS AT EQUAL RECALL@100 (interpolated, unlimited cache); - = not reached ===")
for n in D:
    print(n)
    print("  recall |"+"".join(" %6dKB"%c for c in Cs))
    for tg in (0.3,0.5,0.7,0.9):
        out=[]
        for C in Cs:
            q=qps_at(rows("all_%s_C%dkb.jsonl"%(n,C)),tg)
            out.append("%8s"%("-" if q is None else "%.0f"%q))
        print("  %.1f    |"%tg+"  ".join(out))
print()
print("=== MEMORY LIMIT AT 512 KB: warm QPS (x of unlimited) and cold QPS ===")
for n in D:
    base={r["search_exp"]:r for r in rows("all_%s_C512kb.jsonl"%n)}
    print(n)
    for fr in ("1.0","0.5","0.25","0.1"):
        out=[]
        for r in rows("all_%s_C512kb_m%s.jsonl"%(n,fr)):
            b=base[r["search_exp"]]
            out.append("se%d(%.0fMB) %.1f (%.2fx) cold %.1f usr/q %.0fms"%(r["search_exp"],r["search_exp"]*512/1024.0,warm(r),warm(r)/warm(b),cold(r),r["passes"][-1]["user"]/r["n_queries"]*1000))
        print("  cache %5s: "%fr+" | ".join(out))
print()
print("=== 25% CACHE AT EVERY CLUSTER SIZE: warm QPS as share of unlimited (same search_exp) ===")
for n in D:
    print(n)
    for C in Cs:
        base={r["search_exp"]:r for r in rows("all_%s_C%dkb.jsonl"%(n,C))}
        out=[]
        for r in rows("all_%s_C%dkb_m0.25.jsonl"%(n,C)):
            b=base[r["search_exp"]]
            out.append("se%d: %.1f vs %.1f = %.0f%%"%(r["search_exp"],warm(r),warm(b),100*warm(r)/warm(b)))
        if out: print("  C=%4dKB  "%C+" | ".join(out))
