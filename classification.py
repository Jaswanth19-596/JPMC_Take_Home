import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

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

print("Loading data")
df = pd.read_csv('census-bureau.data', names=cols, na_values=['?', ' ?'], skipinitialspace=True)
# print(df.head())
# print(df.shape)

print("Preprocesing data")
df['label'] = df['label'].astype(str).str.strip().apply(lambda x: 1 if x == '50000+.' else 0)

weights = df['weight'].values
y = df['label'].values
X = df.drop(columns=['label', 'weight'])

cat_cols = X.select_dtypes(include=['object', 'bool']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

X[cat_cols] = X[cat_cols].fillna('Unknown')
X[num_cols] = X[num_cols].fillna(0)

print("Encoding categoricals")
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("Splitting data")
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_encoded, y, weights, test_size=0.2, random_state=42, stratify=y
)

# standard scaler on train and test
print("Scaling")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest Classifier")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_scaled, y_train)

print("Evaluating model")
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.4f}")

with open('classification_report.txt', 'w') as f:
    f.write(report)
    f.write(f"\nROC-AUC: {auc:.4f}\n")

print("Saving visualizations")
# plot ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'RF (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.savefig('classification_roc.png')
plt.close()

# feature importances
importances = rf.feature_importances_
features = X_encoded.columns

# get top 20
idx = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(10, 8))
plt.barh(range(len(idx)), importances[idx], align='center')
plt.yticks(range(len(idx)), [features[i] for i in idx])
plt.gca().invert_yaxis()
plt.title('Top 20 Features')
plt.tight_layout()
plt.savefig('classification_feature_importance.png')
plt.close()

