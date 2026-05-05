import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

cols = [
    "age", "class_of_worker", "detailed_industry_recode", "detailed_occupation_recode",
    "education", "wage_per_hour", "enroll_in_edu_inst_last_wk", "marital_stat",
    "major_industry_code", "major_occupation_code", "race", "hispanic_origin",
    "sex", "member_of_a_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat",
    "capital_gains", "capital_losses", "dividends_from_stocks", "tax_filer_stat",
    "region_of_previous_residence", "state_of_previous_residence", "detailed_household_and_family_stat",
    "detailed_household_summary_in_household", "weight", "migration_code_change_in_msa",
    "migration_code_change_in_reg", "migration_code_move_within_reg", "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer", "family_members_under_18",
    "country_of_birth_father", "country_of_birth_mother", "country_of_birth_self",
    "citizenship", "own_business_or_self_employed", "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits", "weeks_worked_in_year", "year", "label"
]

print("loading data")

df = pd.read_csv('census-bureau.data', names=cols, na_values=['?', ' ?'], skipinitialspace=True)

print(f"shape: {df.shape}")

X = df.drop(columns=['label', 'weight'])

cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X[cat_cols] = X[cat_cols].fillna('Unknown')
X[num_cols] = X[num_cols].fillna(0)

print("encoding")
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enc)


n_clusters = 4
print(f"running kmeans with k={n_clusters}")
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
labels = km.fit_predict(X_scaled)

df['cluster'] = labels

print("\ncluster sizes:")
print(df['cluster'].value_counts())

summary = df.groupby('cluster')[['age', 'wage_per_hour', 'capital_gains', 'weeks_worked_in_year']].mean()
print("\ncluster means:")

print(summary)

edu_mode = df.groupby('cluster')['education'].agg(lambda x: x.mode()[0])
occ_mode = df.groupby('cluster')['major_occupation_code'].agg(lambda x: x.mode()[0])

print("\ntop education per cluster:")
print(edu_mode)
print("\ntop occupation per cluster:")
print(occ_mode)

with open('segmentation_report.txt', 'w') as f:
    f.write("Cluster Sizes:\n")
    f.write(str(df['cluster'].value_counts()) + "\n\n")
    f.write("Cluster Summary (Numeric Means):\n")
    f.write(str(summary) + "\n\n")
    f.write("Top Education Level per Cluster:\n")
    f.write(str(edu_mode) + "\n\n")
    f.write("Top Occupation per Cluster:\n")
    f.write(str(occ_mode) + "\n\n")

print("reducing dims for visualization")
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_scaled)

np.random.seed(42)
sample_idx = np.random.choice(len(X_2d), size=min(10000, len(X_2d)), replace=False)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[sample_idx, 0], X_2d[sample_idx, 1], 
                       c=labels[sample_idx], cmap='viridis', s=8, alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Segments (SVD projection)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('segmentation_svd.png', dpi=150)
plt.close()

