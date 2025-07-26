import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("COVID clinical trials.csv")

# Clean date and extract year
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['Start Year'] = df['Start Date'].dt.year

# Set plot style
sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, y='Status', order=df['Status'].value_counts().index, palette='muted')
plt.title("Trial Status Distribution")
plt.xlabel("Count")
plt.ylabel("Status")
plt.tight_layout()
plt.show()

df['Simple Study Type'] = df['Study Type'].apply(lambda x: x.split(':')[0] if isinstance(x, str) else x)
study_type_simplified = df['Simple Study Type'].value_counts().reset_index()
study_type_simplified.columns = ['Study Type', 'Count']

plt.figure(figsize=(8, 4))
sns.barplot(data=study_type_simplified, y='Study Type', x='Count', palette='pastel')
plt.title("Simplified Study Type Distribution")
plt.xlabel("Count")
plt.ylabel("Study Type")
plt.tight_layout()
plt.show()

top_conditions = df['Conditions'].value_counts().reset_index().head(10)
top_conditions.columns = ['Condition', 'Count']

plt.figure(figsize=(8, 4))
sns.barplot(data=top_conditions, x='Count', y='Condition', palette='Blues_d')
plt.title("Top 10 Conditions Studied")
plt.xlabel("Count")
plt.ylabel("Condition")
plt.tight_layout()
plt.show()

top_interventions = df['Interventions'].dropna().str.split('|').explode().value_counts().reset_index().head(10)
top_interventions.columns = ['Intervention', 'Count']

plt.figure(figsize=(8, 4))
sns.barplot(data=top_interventions, x='Count', y='Intervention', palette='Greens_d')
plt.title("Top 10 Interventions Used")
plt.xlabel("Count")
plt.ylabel("Intervention")
plt.tight_layout()
plt.show()

trial_trend = df['Start Year'].value_counts().sort_index().reset_index()
trial_trend.columns = ['Year', 'Number of Trials']

plt.figure(figsize=(8, 5))
sns.barplot(data=trial_trend, x='Year', y='Number of Trials', palette='Oranges_d')
plt.title("Number of Trials Started Each Year")
plt.xlabel("Year")
plt.ylabel("Number of Trials")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
