import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Load tracking evidence
# -----------------------------------

csv_path = "output/tracking_evidence_v2.csv"

df = pd.read_csv(csv_path)

print("CSV loaded successfully!")
print(f"Total detections: {len(df)}")
print(f"Unique tracks: {df['track_id'].nunique()}")

# -----------------------------------
# 2. Create track summary
# -----------------------------------

track_summary = (
    df.groupby(["camera_id", "track_id"])
    .agg(
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        observations=("track_id", "count"),
        avg_confidence=("confidence", "mean")
    )
    .reset_index()
)

# Calculate duration
track_summary["duration"] = (
    track_summary["last_seen"] -
    track_summary["first_seen"]
)

print("\nTrack Summary:")
print(track_summary)

# -----------------------------------
# 3. Save summary
# -----------------------------------

summary_path = "output/track_summary_v2.csv"

track_summary.to_csv(
    summary_path,
    index=False
)

print(f"\nSummary saved to: {summary_path}")

# -----------------------------------
# 4. Create timeline visualization
# -----------------------------------

plt.figure(figsize=(12, 6))

for _, row in track_summary.iterrows():

    track_id = row["track_id"]
    start = row["first_seen"]
    duration = row["duration"]

    plt.barh(
        track_id,
        duration,
        left=start,
        height=0.5
    )

# -----------------------------------
# 5. Labels
# -----------------------------------

plt.xlabel("Time (seconds)")
plt.ylabel("Track ID")

plt.title(
    "CASELINE - Person Track Timeline"
)

plt.yticks(
    track_summary["track_id"].unique()
)

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

# -----------------------------------
# 6. Save graph
# -----------------------------------

timeline_path = "output/track_timeline.png"

plt.savefig(
    timeline_path,
    dpi=200
)

plt.show()

print(f"Timeline saved to: {timeline_path}")