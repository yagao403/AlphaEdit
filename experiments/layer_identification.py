import numpy as np
import glob

all_scores = []
for f in glob.glob("results/ns3_r0_Qwen_Qwen3-32B/causal_trace/cases/*_mlp.npz"):
    data = np.load(f, allow_pickle=True)
    if not data["correct_prediction"]:
        continue
    scores = data["scores"]
    subject_range = data["subject_range"]
    # Extract the row for the last subject token
    last_subj_idx = subject_range[1] - 1
    all_scores.append(scores[last_subj_idx])

avg_by_layer = np.mean(all_scores, axis=0)
critical_layers = np.argsort(avg_by_layer)[::-1][:10]
print("most important MLP layer centers:", critical_layers)
print("Scores:", avg_by_layer[critical_layers])

