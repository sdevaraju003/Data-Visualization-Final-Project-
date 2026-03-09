import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ai_job_dataset.csv")

# Convert posting date to datetime and create monthly period
df["posting_date"] = pd.to_datetime(df["posting_date"])
df["month"] = df["posting_date"].dt.to_period("M").dt.to_timestamp()

# ---------------------------------
# Visualization 1: AI job postings by month
# ---------------------------------
jobs_month = df.groupby("month").size().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(jobs_month.index, jobs_month.values, linewidth=2)
ax.set_title("AI Job Postings by Month (2024–2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Postings")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ai_jobs_by_month.png", dpi=300)
plt.show()

# ---------------------------------
# Visualization 2: Average salary by experience level
# ---------------------------------
exp_order = ["EN", "MI", "SE", "EX"]
exp_labels = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}

avg_salary = (
    df.groupby("experience_level")["salary_usd"]
    .mean()
    .reindex(exp_order)
)
avg_salary.index = [exp_labels.get(x, x) for x in avg_salary.index]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(avg_salary.index, avg_salary.values)
ax.set_title("Average Salary by Experience Level")
ax.set_xlabel("Average Salary (USD)")
ax.set_ylabel("Experience Level")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("avg_salary_experience.png", dpi=300)
plt.show()

# ---------------------------------
# Visualization 3: Top 10 required AI skills
# ---------------------------------
skills = df["required_skills"].str.split(",", expand=True).stack().str.strip()
top_skills = skills.value_counts().head(10).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(top_skills.index, top_skills.values)
ax.set_title("Top 10 Required AI Skills (Sorted)")
ax.set_xlabel("Skill")
ax.set_ylabel("Job Postings")
ax.grid(True, axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_ai_skills.png", dpi=300)
plt.show()
