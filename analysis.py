import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv('benchmark_data.csv')
le = LabelEncoder()
df['gpu_id'] = le.fit_transform(df['gpu_name'])
prec_map = {'fp32': 0, 'bf16': 1, 'fp16': 2}
df['precision_id'] = df['precision'].map(prec_map)
df['log_compute_load'] = np.log1p((df['model_params'] * df['train_steps']) / df['batch_size'])
feat_cols = ['gpu_id', 'log_compute_load', 'vram_gb', 'precision_id', 'batch_size']
scaler = StandardScaler()
X = scaler.fit_transform(df[feat_cols])
y = np.log1p(df['runtime_sec'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                   subsample=0.7, min_samples_leaf=8, max_features=0.8)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
ape = np.abs(y_pred - y_test_orig) / y_test_orig

print('=== Feature Importances ===')
for feat, imp in zip(feat_cols, model.feature_importances_):
    print(f'  {feat}: {imp:.4f}')

print()
print('=== Overall Metrics ===')
print(f'  MAPE: {mean_absolute_percentage_error(y_test_orig, y_pred):.4f}')
print(f'  Median APE: {np.median(ape):.4f}')
print(f'  APE < 5%: {(ape<0.05).sum()} / {len(ape)} = {(ape<0.05).mean():.1%}')
print(f'  APE < 10%: {(ape<0.10).sum()} / {len(ape)} = {(ape<0.10).mean():.1%}')

print()
print('=== APE Percentiles ===')
for p in [25, 50, 75, 90, 95]:
    print(f'  p{p}: {np.percentile(ape, p):.4f}')
