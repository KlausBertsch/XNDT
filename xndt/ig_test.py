"""Final, frozen 512-interval Integrated Gradients evaluation on all test scenarios."""
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np, tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xndt.model import RouteNet_Fermi
from xndt.data import DEMAND_ORDER, adapt_sample, read_jsonl
from xndt.ig_convergence import with_composite_rates
from xndt.integrated_gradients import (COMPLETENESS_TOLERANCE,
 FINAL_IG_STEPS, occlusion, output, ranking, seed_all, select_low_load,
 traffic_and_ratio, validate_inputs, weight_fingerprint)

assert FINAL_IG_STEPS==512

def arguments():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset-dir",required=True,type=Path); p.add_argument("--training-dir",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path); p.add_argument("--seed",type=int,default=42); return p.parse_args()
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()
def write_csv(path,data):
 with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def select_test_representatives(dataset_dir):
 raw=list(read_jsonl(dataset_dir/"test.jsonl")); groups=defaultdict(list)
 for row in raw: groups[row["scenario_key"]].append(row)
 if len(raw)!=42 or len(groups)!=14 or any(len(v)!=3 for v in groups.values()): raise ValueError("Expected 42 test runs, 14 scenarios and three repetitions")
 return [min(v,key=lambda x:(x["repetition_index"],x["run_id"])) for _,v in sorted(groups.items())]
@tf.function(reduce_retracing=True)
def gradient_grid_512(model,inputs,baseline,actual,ratios,target_index):
 alphas=tf.linspace(0.,1.,FINAL_IG_STEPS+1)
 def one(alpha):
  rates=baseline+alpha*(actual-baseline)
  with tf.GradientTape() as tape:
   tape.watch(rates); prediction=model(with_composite_rates(inputs,rates,ratios),training=False)[target_index,0]
  return tape.gradient(prediction,rates)
 return tf.map_fn(one,alphas,fn_output_signature=tf.TensorSpec((3,),tf.float32))
def integrate(grid,actual,baseline):
 average=(grid[0]+grid[-1]+2*tf.reduce_sum(grid[1:-1],axis=0))/(2*FINAL_IG_STEPS); return (actual-baseline)*average
def top_forward_set(values): return {DEMAND_ORDER[i] for i in np.argsort(-np.abs(values))[:2]}=={"d1","d2"}
def plots(out,rows,baseline_pairs):
 for target in ("d1","d2"):
  for measure,label,prefix in (("absolute_ig","Absolute IG","absolute"),("relative_importance","Relative |IG|","relative")):
   fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
   for ax,baseline in zip(axes,("zero_traffic","low_load")):
    selected=[x for x in rows if x["target_demand"]==target and x["baseline"]==baseline]
    for demand in DEMAND_ORDER:
     values=sorted([x for x in selected if x["attributed_demand"]==demand],key=lambda x:x["rho_wire"]); ax.plot([x["rho_wire"] for x in values],[x[measure] for x in values],marker="o",label=demand)
    ax.set(title=baseline,xlabel="rho_wire",ylabel=label); ax.legend()
   fig.tight_layout(); fig.savefig(out/f"test_ig_{prefix}_vs_rho_{target}_target.png",dpi=150); plt.close(fig)
 fig,ax=plt.subplots(); ax.scatter([x["relative_d1_difference"] for x in baseline_pairs],[x["relative_d2_difference"] for x in baseline_pairs],c=[x["rho_wire"] for x in baseline_pairs]); ax.set(xlabel="|Δ relative d1|",ylabel="|Δ relative d2|"); fig.tight_layout(); fig.savefig(out/"test_ig_baseline_comparison.png",dpi=150); plt.close(fig)
def main():
 args=arguments(); seed_all(args.seed); checkpoint=args.training_dir/"best.weights.h5"; normalization_path=args.training_dir/"xndt_normalization.json"; validation=list(read_jsonl(args.dataset_dir/"validation.jsonl")); selected=select_test_representatives(args.dataset_dir); normalization=json.loads(normalization_path.read_text()); model=RouteNet_Fermi(xndt_mode=True,normalization=normalization["features"]); first,_,_=adapt_sample(selected[0],True); model(first,training=False); model.load_weights(checkpoint); before=weight_fingerprint(model); checkpoint_hash=sha256(checkpoint); rows=[]; completeness=[]; occlusion_rows=[]; explanation_values={}
 for raw in selected:
  inputs,_,_=adapt_sample(raw,True); validate_inputs(inputs); actual,ratios=traffic_and_ratio(inputs); low=select_low_load(validation,raw); low_inputs,_,_=adapt_sample(low,True); low_rates,_=traffic_and_ratio(low_inputs)
  for target_index,target in enumerate(("d1","d2")):
   for baseline_name,baseline in (("zero_traffic",tf.zeros_like(actual)),("low_load",low_rates)):
    grid=gradient_grid_512(model,inputs,tf.cast(baseline,tf.float32),actual,ratios,target_index)
    if not bool(tf.reduce_all(tf.math.is_finite(grid))): raise ValueError("Non-finite gradient")
    attrs=np.asarray(integrate(grid,actual,baseline)); p_actual=float(output(model,inputs,actual,ratios,target_index)); p_baseline=float(output(model,inputs,baseline,ratios,target_index)); difference=p_actual-p_baseline; error=abs(float(attrs.sum())-difference); denominator=float(np.abs(attrs).sum()); relative=np.abs(attrs)/denominator if denominator else np.zeros(3); deltas=np.asarray(occlusion(model,inputs,baseline,target_index)); key=(raw["scenario_key"],target,baseline_name); explanation_values[key]=(attrs,relative,deltas); meta=raw["metadata"]
    completeness.append({"scenario_key":raw["scenario_key"],"rho_wire":meta["rho_wire"],"target_demand":target,"baseline":baseline_name,"prediction_actual":p_actual,"prediction_baseline":p_baseline,"sum_ig":float(attrs.sum()),"prediction_difference":difference,"completeness_error":error,"passes_completeness":error<=COMPLETENESS_TOLERANCE})
    occlusion_rows.append({"scenario_key":raw["scenario_key"],"rho_wire":meta["rho_wire"],"target_demand":target,"baseline":baseline_name,"ig_ranking":ranking(attrs),"occlusion_ranking":ranking(deltas),"exact_ranking_agreement":ranking(attrs)==ranking(deltas),"ig_d3_last":ranking(attrs).endswith("d3"),"occlusion_d3_last":ranking(deltas).endswith("d3"),"ig_forward_set":top_forward_set(attrs),"occlusion_forward_set":top_forward_set(deltas)})
    for i,demand in enumerate(DEMAND_ORDER): rows.append({"scenario_key":raw["scenario_key"],"rho_wire":meta["rho_wire"],"alpha":meta["alpha"],"control_wire_rate_gbps":meta["control_wire_rate_gbps"],"target_demand":target,"baseline":baseline_name,"attributed_demand":demand,"prediction_actual":p_actual,"prediction_baseline":p_baseline,"signed_ig":float(attrs[i]),"absolute_ig":float(abs(attrs[i])),"relative_importance":float(relative[i]),"completeness_error":error,"passes_completeness":error<=COMPLETENESS_TOLERANCE})
 baseline_pairs=[]
 for raw in selected:
  for target in ("d1","d2"):
   zero=explanation_values[(raw["scenario_key"],target,"zero_traffic")]; low=explanation_values[(raw["scenario_key"],target,"low_load")]; baseline_pairs.append({"scenario_key":raw["scenario_key"],"rho_wire":raw["metadata"]["rho_wire"],"target_demand":target,"zero_ranking":ranking(zero[0]),"low_load_ranking":ranking(low[0]),"ranking_agreement":ranking(zero[0])==ranking(low[0]),"zero_d3_last":ranking(zero[0]).endswith("d3"),"low_load_d3_last":ranking(low[0]).endswith("d3"),"zero_forward_set":top_forward_set(zero[0]),"low_load_forward_set":top_forward_set(low[0]),"forward_set_agreement":top_forward_set(zero[0])==top_forward_set(low[0]),"relative_d1_difference":float(abs(zero[1][0]-low[1][0])),"relative_d2_difference":float(abs(zero[1][1]-low[1][1])),"signed_d1_difference":float(abs(zero[0][0]-low[0][0])),"signed_d2_difference":float(abs(zero[0][1]-low[0][1])),"signed_d3_difference":float(abs(zero[0][2]-low[0][2]))})
 after=weight_fingerprint(model)
 if before!=after: raise AssertionError("Weights changed during final test XAI")
 errors=np.array([x["completeness_error"] for x in completeness]); congested=[x for x in rows if x["rho_wire"]>1]; attr_summary={}
 for d in DEMAND_ORDER:
  values=[x for x in congested if x["attributed_demand"]==d]; attr_summary[d]={"mean_absolute_ig":float(np.mean([x["absolute_ig"] for x in values])),"mean_relative_importance":float(np.mean([x["relative_importance"] for x in values]))}
 explanations=defaultdict(list)
 for x in congested: explanations[(x["scenario_key"],x["target_demand"],x["baseline"])].append(x)
 forward=[sum(x["relative_importance"] for x in v if x["attributed_demand"] in ("d1","d2")) for v in explanations.values()]; control=[sum(x["relative_importance"] for x in v if x["attributed_demand"]=="d3") for v in explanations.values()]; d3=[x["absolute_ig"] for x in rows if x["attributed_demand"]=="d3"]
 summary={"protocol":{"method":"Integrated Gradients","steps":512,"points":513,"integration":"uniform trapezoidal","tolerance":COMPLETENESS_TOLERANCE},"checkpoint_sha256":checkpoint_hash,"weight_fingerprint_before":before,"weight_fingerprint_after":after,"weights_unchanged":before==after,"test_scenarios":14,"explanations":56,"completeness":{"n_pass":int(np.sum(errors<=COMPLETENESS_TOLERANCE)),"n_fail":int(np.sum(errors>COMPLETENESS_TOLERANCE)),"mean_error":float(errors.mean()),"median_error":float(np.median(errors)),"max_error":float(errors.max())},"congested_attribution":attr_summary,"mean_forward_relevance":float(np.mean(forward)),"max_control_relevance":float(np.max(control)),"d3_negative_control":{"mean_absolute_ig":float(np.mean(d3)),"max_absolute_ig":float(np.max(d3))},"baseline_robustness":{"pairs":len(baseline_pairs),"ranking_agreement":sum(x["ranking_agreement"] for x in baseline_pairs),"d3_last_both":sum(x["zero_d3_last"] and x["low_load_d3_last"] for x in baseline_pairs),"forward_set_agreement":sum(x["forward_set_agreement"] for x in baseline_pairs),"forward_set_both":sum(x["zero_forward_set"] and x["low_load_forward_set"] for x in baseline_pairs)},"occlusion":{"comparisons":len(occlusion_rows),"exact_ranking_agreement":sum(x["exact_ranking_agreement"] for x in occlusion_rows),"d3_last_both":sum(x["ig_d3_last"] and x["occlusion_d3_last"] for x in occlusion_rows),"forward_demand_agreement":sum(x["ig_forward_set"] and x["occlusion_forward_set"] for x in occlusion_rows)},"all_finite":bool(np.isfinite([x["signed_ig"] for x in rows]).all()),"repetition_rule":"minimum repetition_index, then run_id","low_load_rule":"validation rho=0.20 loss-free; minimum alpha distance, control-rate distance, scenario_key/run_id"}
 args.output_dir.mkdir(parents=True,exist_ok=True); write_csv(args.output_dir/"ig_test.csv",rows); write_csv(args.output_dir/"ig_test_completeness.csv",completeness); write_csv(args.output_dir/"ig_test_baseline_comparison.csv",baseline_pairs); write_csv(args.output_dir/"ig_test_occlusion_comparison.csv",occlusion_rows); (args.output_dir/"ig_test.json").write_text(json.dumps({"summary":summary,"attributions":rows},indent=2,sort_keys=True)+"\n"); (args.output_dir/"ig_test_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); plots(args.output_dir,rows,baseline_pairs); print(json.dumps(summary,sort_keys=True),flush=True)
if __name__=="__main__": main()
