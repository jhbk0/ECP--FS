import json,os,sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA,OUT=sys.argv[1],sys.argv[2]
INK="#1A1A1A"
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":8,
 "axes.edgecolor":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"text.color":INK,
 "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"pdf.fonttype":42})
Cs=[128,256,512,1024,2048,4096,8192]
col=["#0072B2","#E69F00","#009E73","#CC79A7","#D55E00","#56B4E9","#222222"]
mk=["o","s","^","D","v","P","X"]
def rows(f):
    p=os.path.join(DATA,f); return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def lab(C): return "%d KB"%C if C<1024 else "%d MB"%(C//1024)
def style(ax):
    for s in("top","right"): ax.spines[s].set_visible(False)
    ax.grid(color="k",alpha=.08,lw=.5); ax.set_axisbelow(True)
names={"ccnews":"ccnews","imagenet":"imagenet-clip","laion":"laion-clip (OOD)","msmarco":"msmarco"}
fig,axs=plt.subplots(2,2,figsize=(7.0,4.6),dpi=300)
for ax,n,t in zip(axs.flat,["ccnews","imagenet","laion","msmarco"],"abcd"):
    for C,c,m in zip(Cs,col,mk):
        rs=sorted(rows("all_%s_C%dkb.jsonl"%(n,C)),key=lambda r:r["recall"])
        if rs: ax.plot([r["recall"] for r in rs],[r["passes"][-1]["qps"] for r in rs],marker=m,ms=3.6,lw=.9,color=c,label=lab(C),mec="white",mew=.3)
    ax.set_yscale("log"); ax.set_xlim(0,1.0); style(ax)
    ax.set_title("(%s) %s"%(t,names[n]),loc="left",fontsize=8.5)
    ax.set_xlabel("Recall@100"); ax.set_ylabel("Queries per second (warm)")
h,l=axs[0,0].get_legend_handles_labels()
fig.legend(h,l,title="Cluster size",ncol=7,frameon=False,loc="lower center",fontsize=8,title_fontsize=8,handlelength=1.8,columnspacing=1.2)
fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(os.path.join(OUT,"fig_cluster_size.pdf")); plt.close(fig)

# fig 3: (a) cache share at 512KB, B=50MB ; (b) warm/cold ratio vs budget per dataset, 512KB
D=["ccnews","imagenet","laion","msmarco"]; dc=["#0072B2","#E69F00","#009E73","#D55E00"]; dm=["o","s","^","D"]
fr=["1.0","0.5","0.25","0.1"]; xl=["unl.","100%","50%","25%","10%"]
fig,axs=plt.subplots(1,2,figsize=(7.0,2.7),dpi=300)
for n,c,m in zip(D,dc,dm):
    base={r["search_exp"]:r for r in rows("all_%s_C512kb.jsonl"%n)}
    b=base[100]; share=[100.]
    for f in fr:
        r=[x for x in rows("all_%s_C512kb_m%s.jsonl"%(n,f)) if x["search_exp"]==100][0]
        share.append(100*r["passes"][-1]["qps"]/b["passes"][-1]["qps"])
    axs[0].plot(range(5),share,marker=m,ms=4,lw=1,color=c,label=names[n],mec="white",mew=.3)
    ses=sorted(base); xs=[s*512/1024 for s in ses if s>=6]
    ys=[base[s]["passes"][-1]["qps"]/base[s]["passes"][0]["qps"] for s in ses if s>=6]
    axs[1].plot(xs,ys,marker=m,ms=4,lw=1,color=c,mec="white",mew=.3)
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(xl); axs[0].set_ylim(0,108)
axs[0].set_xlabel("Cache limit (share of index size on disk)"); axs[0].set_ylabel("Warm speed (% of unlimited)")
axs[0].set_title("(a) Limited cache, 512 KB clusters, 50 MB per query",loc="left",fontsize=8.5)
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_yticks([1,2,5,10,20,40]); axs[1].set_yticklabels(["1","2","5","10","20","40"]); axs[1].minorticks_off()
axs[1].set_xlabel("Nominal data opened per query (MB)"); axs[1].set_ylabel("Warm / cold speed")
axs[1].set_title("(b) Cold start, 512 KB clusters",loc="left",fontsize=8.5)
for ax in axs: style(ax)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,ncol=4,frameon=False,loc="lower center",fontsize=8,handlelength=1.8,columnspacing=1.6)
fig.tight_layout(rect=(0,.09,1,1)); fig.savefig(os.path.join(OUT,"fig_memory_cold.pdf")); plt.close(fig)
