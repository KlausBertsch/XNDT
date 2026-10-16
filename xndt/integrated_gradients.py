"""Validation-only Integrated Gradients protocol for frozen XNDT traffic inputs."""
import argparse, csv, hashlib, json, os, random
from pathlib import Path
import numpy as np, tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xndt.model import RouteNet_Fermi
from xndt.data import DEMAND_ORDER, adapt_sample, read_jsonl

RHO_LEVELS=(.60,.95,1.02,1.10,1.20,1.40)
FORBIDDEN={"rho_wire","alpha","control_wire_rate_gbps","loss_ratio","tx_packets","rx_packets","loss_packets","forward_loss_ratio","qfx_drops","queue_occupancy","measured_utilization","telemetry"}
COMPLETENESS_TOLERANCE=1.2e-3
FINAL_IG_STEPS=512

def arguments():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset-dir",required=True,type=Path); p.add_argument("--training-dir",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path); p.add_argument("--seed",type=int,default=42); p.add_argument("--steps",type=int,default=FINAL_IG_STEPS); return p.parse_args()
def seed_all(seed): os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed); tf.config.experimental.enable_op_determinism()
def write_csv(path,data):
 with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def weight_fingerprint(model):
 h=hashlib.sha256()
 for value in model.weights: h.update(value.path.encode()); h.update(value.numpy().tobytes())
 return h.hexdigest()
def validate_inputs(inputs):
 found=FORBIDDEN&set(inputs)
 if found: raise ValueError(f"Forbidden gradient-path inputs: {sorted(found)}")
 if inputs["link_to_path"].numpy().tolist()[2][1]==inputs["link_to_path"].numpy().tolist()[0][1]: raise ValueError("d3 incorrectly shares forward bottleneck link entity")
def validation_scenarios(dataset_dir):
 raw=list(read_jsonl(dataset_dir/"validation.jsonl")); by_rho={}
 for row in sorted(raw,key=lambda x:(x["scenario_key"],x["run_id"])): by_rho.setdefault(float(row["metadata"]["rho_wire"]),row)
 missing=[rho for rho in RHO_LEVELS if rho not in by_rho]
 if missing: raise ValueError(f"Missing validation rho levels: {missing}")
 return raw,[by_rho[rho] for rho in RHO_LEVELS]
def select_low_load(raw,explained):
 candidates=[x for x in raw if float(x["metadata"]["rho_wire"])==.2 and all(d["label"]["loss_ratio"]==0 for d in x["demands"])]
 if not candidates: raise ValueError("No loss-free rho=0.20 validation reference")
 meta=explained["metadata"]
 return min(candidates,key=lambda x:(abs(x["metadata"]["alpha"]-meta["alpha"]),abs(x["metadata"]["control_wire_rate_gbps"]-meta["control_wire_rate_gbps"]),x["scenario_key"],x["run_id"]))
def traffic_and_ratio(inputs):
 traffic=tf.reshape(tf.cast(inputs["traffic"],tf.float32),(-1,)); packets=tf.reshape(tf.cast(inputs["packets"],tf.float32),(-1,)); return traffic,tf.math.divide_no_nan(packets,traffic)
def with_rates(inputs,rates,ratios):
 result=dict(inputs); result["traffic"]=tf.reshape(rates,(-1,1)); result["packets"]=tf.reshape(rates*ratios,(-1,1)); return result
def output(model,inputs,rates,ratios,target_index): return model(with_rates(inputs,rates,ratios),training=False)[target_index,0]
@tf.function(reduce_retracing=True)
def rate_gradient(model,inputs,rates,ratios,target_index):
 with tf.GradientTape() as tape:
  tape.watch(rates); prediction=output(model,inputs,rates,ratios,target_index)
 return prediction,tape.gradient(prediction,rates)

def integrated_gradients(model,inputs,baseline_rates,target_index,steps):
 if steps<=0: raise ValueError("steps must be positive")
 actual,ratios=traffic_and_ratio(inputs); baseline=tf.cast(baseline_rates,tf.float32); gradients=[]
 for alpha in tf.linspace(0.,1.,steps+1):
  rates=baseline+alpha*(actual-baseline)
  prediction,gradient=rate_gradient(model,inputs,rates,ratios,target_index)
  if gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient))): raise ValueError("Invalid IG gradient")
  gradients.append(gradient)
 gradients=tf.stack(gradients); trapezoid=(gradients[0]+gradients[-1]+2*tf.reduce_sum(gradients[1:-1],axis=0))/(2*steps); attribution=(actual-baseline)*trapezoid; actual_prediction=output(model,inputs,actual,ratios,target_index); baseline_prediction=output(model,inputs,baseline,ratios,target_index); error=tf.abs(tf.reduce_sum(attribution)-(actual_prediction-baseline_prediction))
 return np.asarray(attribution),float(actual_prediction),float(baseline_prediction),float(error)
def occlusion(model,inputs,baseline_rates,target_index):
 actual,ratios=traffic_and_ratio(inputs); original=float(output(model,inputs,actual,ratios,target_index)); deltas=[]
 for i in range(3):
  rates=tf.tensor_scatter_nd_update(actual,[[i]],[baseline_rates[i]]); deltas.append(original-float(output(model,inputs,rates,ratios,target_index)))
 return deltas
def ranking(values): return ">".join(DEMAND_ORDER[i] for i in np.argsort(-np.abs(values)))
def make_plots(out,rows):
 for target in ("d1","d2"):
  fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
  for ax,baseline in zip(axes,("zero_traffic","low_load")):
   selected=[x for x in rows if x["target_demand"]==target and x["baseline"]==baseline]
   for demand in DEMAND_ORDER:
    values=[x for x in selected if x["attributed_demand"]==demand]; ax.plot([x["rho_wire"] for x in values],[x["relative_importance"] for x in values],marker="o",label=demand)
   ax.set(title=baseline,xlabel="rho_wire",ylabel="Relative |IG|"); ax.legend()
  fig.tight_layout(); fig.savefig(out/f"ig_by_rho_{target}_target.png",dpi=150); plt.close(fig)
 fig,ax=plt.subplots()
 for demand in DEMAND_ORDER:
  selected=[x for x in rows if x["attributed_demand"]==demand]; ax.scatter([x["signed_ig"] for x in selected],[x["occlusion_delta"] for x in selected],s=18,alpha=.65,label=demand)
 ax.set(xlabel="Signed IG",ylabel="Occlusion delta"); ax.legend(); fig.tight_layout(); fig.savefig(out/"ig_vs_occlusion.png",dpi=150); plt.close(fig)
def main():
 args=arguments(); seed_all(args.seed); checkpoint=args.training_dir/"best.weights.h5"; normalization_path=args.training_dir/"xndt_normalization.json"
 for path in (checkpoint,normalization_path,args.dataset_dir/"validation.jsonl"):
  if not path.is_file(): raise FileNotFoundError(path)
 normalization=json.loads(normalization_path.read_text()); all_validation,selected=validation_scenarios(args.dataset_dir); model=RouteNet_Fermi(xndt_mode=True,normalization=normalization["features"]); first,_,_=adapt_sample(selected[0],True); validate_inputs(first); model(first,training=False); model.load_weights(checkpoint); before=weight_fingerprint(model); rows=[]; completeness=[]; comparisons=[]; baseline_meta={}
 for raw in selected:
  inputs,_,_=adapt_sample(raw,True); validate_inputs(inputs); actual,_=traffic_and_ratio(inputs); low=select_low_load(all_validation,raw); low_inputs,_,_=adapt_sample(low,True); low_rates,_=traffic_and_ratio(low_inputs); baselines={"zero_traffic":tf.zeros_like(actual),"low_load":low_rates}; baseline_meta[raw["scenario_key"]]={"selection_rule":"loss-free validation rho=0.20; minimize |alpha difference|, then |control-rate difference|, then scenario_key/run_id","low_load_scenario_key":low["scenario_key"],"low_load_run_id":low["run_id"]}
  for target_index,target in enumerate(("d1","d2")):
   for baseline_name,baseline_rates in baselines.items():
    attrs,p_actual,p_baseline,error=integrated_gradients(model,inputs,baseline_rates,target_index,args.steps); deltas=occlusion(model,inputs,baseline_rates,target_index); denominator=float(np.sum(np.abs(attrs))); relative=np.abs(attrs)/denominator if denominator>0 else np.zeros(3); completeness.append({"scenario_key":raw["scenario_key"],"rho_wire":raw["metadata"]["rho_wire"],"target_demand":target,"baseline":baseline_name,"sum_ig":float(attrs.sum()),"prediction_difference":p_actual-p_baseline,"completeness_error":error,"within_tolerance":error<=COMPLETENESS_TOLERANCE}); comparisons.append({"scenario_key":raw["scenario_key"],"rho_wire":raw["metadata"]["rho_wire"],"target_demand":target,"baseline":baseline_name,"ig_ranking":ranking(attrs),"occlusion_ranking":ranking(deltas),"ranking_match":ranking(attrs)==ranking(deltas)})
    for i,demand in enumerate(DEMAND_ORDER): rows.append({"scenario_key":raw["scenario_key"],"rho_wire":raw["metadata"]["rho_wire"],"target_demand":target,"baseline":baseline_name,"attributed_demand":demand,"prediction_actual":p_actual,"prediction_baseline":p_baseline,"signed_ig":float(attrs[i]),"absolute_ig":float(abs(attrs[i])),"relative_importance":float(relative[i]),"occlusion_delta":float(deltas[i]),"completeness_error":error})
 after=weight_fingerprint(model)
 if before!=after: raise AssertionError("Model weights changed during IG")
 if not all(np.isfinite([x["signed_ig"] for x in rows])) or not all(np.isfinite([x["prediction_actual"] for x in rows])): raise ValueError("Non-finite XAI output")
 protocol_valid=all(x["within_tolerance"] for x in completeness)
 args.output_dir.mkdir(parents=True,exist_ok=True); write_csv(args.output_dir/"ig_validation.csv",rows); write_csv(args.output_dir/"ig_completeness.csv",completeness); write_csv(args.output_dir/"ig_occlusion_comparison.csv",comparisons); write_csv(args.output_dir/"ig_baseline_comparison.csv",comparisons); make_plots(args.output_dir,rows); summary={"split":"validation only","steps":args.steps,"integration":"trapezoidal rule over steps+1 points","explained_variables":"per-demand configured wire rate, with packet rate coupled by each sample PPS/BPS ratio","baselines":{"zero_traffic":"all demand traffic and packet rates zero",**baseline_meta},"completeness_tolerance":COMPLETENESS_TOLERANCE,"max_completeness_error":max(x["completeness_error"] for x in completeness),"weight_fingerprint_before":before,"weight_fingerprint_after":after,"weights_unchanged":before==after,"protocol_valid":protocol_valid,"rows":rows,"comparisons":comparisons}; (args.output_dir/"ig_validation.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","baselines")},sort_keys=True),flush=True)
 if not protocol_valid: raise SystemExit("IG PROTOCOL FAIL: completeness tolerance exceeded")
if __name__=="__main__": main()
