import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# === Input and output paths ===
detailed_file = "./CalmidDetailedResults.csv"
output_file = "./Bnechmark/CalmidBenchmark.csv"

# === Load detailed results ===
df = pd.read_csv(detailed_file, header=None)

# Store results
results = []

all_true = []
all_pred = []

# Process each pair of rows (true/predicted for a size)
for i in range(0, len(df), 2):
    size_label = df.iloc[i, 0]   # e.g. "0-true"
    size = size_label.split("-")[0]  # just "0"

    true_labels = df.iloc[i, 1:].values.astype(int)
    pred_labels = df.iloc[i+1, 1:].values.astype(int)

    # Save for overall calculation
    all_true.extend(true_labels)
    all_pred.extend(pred_labels)

    # Calculate metrics
    precision = round(precision_score(true_labels, pred_labels, average="weighted", zero_division=0), 2)
    recall = round(recall_score(true_labels, pred_labels, average="weighted", zero_division=0), 2)
    accuracy = round(accuracy_score(true_labels, pred_labels), 2)
    f1 = round(f1_score(true_labels, pred_labels, average="weighted", zero_division=0), 2)


    results.append([size, precision, recall, accuracy, f1])

# === Compute overall metrics ===
precision = round(precision_score(all_true, all_pred, average="weighted", zero_division=0),2)
recall = round(recall_score(all_true, all_pred, average="weighted", zero_division=0),2)
accuracy = round(accuracy_score(all_true, all_pred),2)
f1 =round( f1_score(all_true, all_pred, average="weighted", zero_division=0) ,2)

results.append(["overall", precision, recall, accuracy, f1])

# === Save to CSV ===
results_df = pd.DataFrame(results, columns=["size", "precision", "recall", "accuracy", "f1_score"])
results_df.to_csv(output_file, index=False)

print(f"Metrics saved to {output_file}")
