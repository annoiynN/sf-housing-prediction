
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')

print("ALGORITHM COMPARISON STUDY")
print("San Francisco Housing Price Prediction")
print("=" * 50)

# 1. DATA LOADING AND PREPARATION
print("\n1. DATA LOADING AND PREPARATION")
print("-" * 30)

# Load dataset
file_path = os.path.expanduser('~/Downloads/archive/sf_clean.csv')
df = pd.read_csv(file_path)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target variable: price (${df['price'].min():.0f} - ${df['price'].max():.0f})")

# Basic data exploration
print(f"\nBasic Statistics:")
print(f"Average price: ${df['price'].mean():.0f}")
print(f"Median price: ${df['price'].median():.0f}")
print(f"Standard deviation: ${df['price'].std():.0f}")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# 2. FEATURE ENGINEERING
print("\n2. FEATURE ENGINEERING")
print("-" * 30)

# Create working copy
df_model = df.copy()

# Encode categorical variables
categorical_features = ['laundry', 'pets', 'housing_type', 'parking']
encoders = {}

print("Encoding categorical features:")
for feature in categorical_features:
    encoder = LabelEncoder()
    df_model[f'{feature}_encoded'] = encoder.fit_transform(df_model[feature])
    encoders[feature] = encoder
    print(f"  {feature}: {len(encoder.classes_)} categories")

# Create new features
print("\nCreating engineered features:")
df_model['price_per_sqft'] = df_model['price'] / df_model['sqft']
df_model['rooms_total'] = df_model['beds'] + df_model['bath']
df_model['sqft_per_room'] = df_model['sqft'] / (df_model['rooms_total'] + 1)

print("  price_per_sqft: Price efficiency metric")
print("  rooms_total: Total number of rooms")
print("  sqft_per_room: Space efficiency metric")

# Select final features
feature_columns = [
    'sqft', 'beds', 'bath', 'hood_district',
    'laundry_encoded', 'pets_encoded', 'housing_type_encoded', 'parking_encoded',
    'rooms_total', 'sqft_per_room'
]

X = df_model[feature_columns]
y = df_model['price']

print(f"\nFinal feature set: {len(feature_columns)} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. ALGORITHM IMPLEMENTATION
print("\n3. ALGORITHM IMPLEMENTATION")
print("-" * 30)

# Define algorithms to compare
algorithms = {
    'Linear Regression': {
        'model': LinearRegression(),
        'use_scaling': False,
        'description': 'Simple linear relationship'
    },
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=100, random_state=42),
        'use_scaling': False,
        'description': 'Ensemble of decision trees'
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'use_scaling': False,
        'description': 'Sequential learning ensemble'
    },
    'Support Vector Regression': {
        'model': SVR(kernel='rbf', C=100, gamma=0.1),
        'use_scaling': True,
        'description': 'Non-linear kernel method'
    }
}

print("Algorithms selected for comparison:")
for name, info in algorithms.items():
    print(f"  {name}: {info['description']}")

# 4. MODEL TRAINING AND EVALUATION
print("\n4. MODEL TRAINING AND EVALUATION")
print("-" * 30)

results = {}

print(f"{'Algorithm':<25} {'R2 Score':<10} {'RMSE':<10} {'MAE':<10}")
print("-" * 55)

for name, config in algorithms.items():
    model = config['model']
    
    # Choose appropriate data
    if config['use_scaling']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred,
        'use_scaling': config['use_scaling']
    }
    
    print(f"{name:<25} {r2:<10.3f} {rmse:<10.0f} {mae:<10.0f}")

# 5. CROSS-VALIDATION
print("\n5. CROSS-VALIDATION ANALYSIS")
print("-" * 30)

cv_results = {}

print(f"{'Algorithm':<25} {'CV Mean':<10} {'CV Std':<10}")
print("-" * 45)

for name, config in algorithms.items():
    model = config['model']
    
    # Perform cross-validation
    if config['use_scaling']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    cv_results[name] = cv_scores
    print(f"{name:<25} {cv_scores.mean():<10.3f} {cv_scores.std():<10.3f}")

# 6. FEATURE IMPORTANCE ANALYSIS
print("\n6. FEATURE IMPORTANCE ANALYSIS")
print("-" * 30)

# Analyze feature importance for tree-based models
tree_models = ['Random Forest', 'Gradient Boosting']

for model_name in tree_models:
    if model_name in results:
        model = results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            print(f"\n{model_name} - Top 5 Most Important Features:")
            
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']:<20} {row['importance']:.3f}")

# 7. VISUALIZATION
print("\n7. RESULTS VISUALIZATION")
print("-" * 30)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Algorithm Comparison Results', fontsize=16)

# Plot 1: R2 Score Comparison
ax1 = axes[0, 0]
models = list(results.keys())
r2_scores = [results[model]['r2'] for model in models]
colors = ['skyblue', 'lightgreen', 'coral', 'gold']

bars = ax1.bar(models, r2_scores, color=colors)
ax1.set_title('R² Score Comparison')
ax1.set_ylabel('R² Score')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# Plot 2: RMSE Comparison
ax2 = axes[0, 1]
rmse_scores = [results[model]['rmse'] for model in models]
bars2 = ax2.bar(models, rmse_scores, color=colors)
ax2.set_title('RMSE Comparison')
ax2.set_ylabel('RMSE ($)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Cross-validation box plot
ax3 = axes[0, 2]
cv_data = [cv_results[model] for model in models]
bp = ax3.boxplot(cv_data, labels=[m.split()[0] for m in models])
ax3.set_title('Cross-Validation Scores')
ax3.set_ylabel('CV R² Score')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Best model predictions vs actual
best_model = max(results.keys(), key=lambda k: results[k]['r2'])
best_predictions = results[best_model]['predictions']

ax4 = axes[1, 0]
ax4.scatter(y_test, best_predictions, alpha=0.6, color='purple')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax4.set_xlabel('Actual Price ($)')
ax4.set_ylabel('Predicted Price ($)')
ax4.set_title(f'Best Model: {best_model}')
ax4.grid(alpha=0.3)

# Plot 5: Feature importance for best tree model
ax5 = axes[1, 1]
best_tree_model = None
for model_name in ['Random Forest', 'Gradient Boosting']:
    if model_name in results and hasattr(results[model_name]['model'], 'feature_importances_'):
        if best_tree_model is None or results[model_name]['r2'] > results[best_tree_model]['r2']:
            best_tree_model = model_name

if best_tree_model:
    model = results[best_tree_model]['model']
    feature_imp = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(8)
    
    ax5.barh(feature_imp['feature'], feature_imp['importance'], color='lightcoral')
    ax5.set_title(f'Feature Importance\n({best_tree_model})')
    ax5.set_xlabel('Importance')
    ax5.grid(axis='x', alpha=0.3)

# Plot 6: Model performance summary
ax6 = axes[1, 2]
performance_metrics = pd.DataFrame({
    'Model': models,
    'R2': r2_scores,
    'RMSE': rmse_scores
})

# Normalize metrics for comparison (0-1 scale)
performance_metrics['R2_norm'] = performance_metrics['R2']
performance_metrics['RMSE_norm'] = 1 - (performance_metrics['RMSE'] - performance_metrics['RMSE'].min()) / (performance_metrics['RMSE'].max() - performance_metrics['RMSE'].min())

x = np.arange(len(models))
width = 0.35

bars1 = ax6.bar(x - width/2, performance_metrics['R2_norm'], width, label='R² Score', alpha=0.8)
bars2 = ax6.bar(x + width/2, performance_metrics['RMSE_norm'], width, label='RMSE (inverted)', alpha=0.8)

ax6.set_xlabel('Models')
ax6.set_ylabel('Normalized Score')
ax6.set_title('Overall Performance Comparison')
ax6.set_xticks(x)
ax6.set_xticklabels([m.split()[0] for m in models])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 8. FINAL COMPARISON AND RECOMMENDATIONS
print("\n8. ALGORITHM COMPARISON SUMMARY")
print("=" * 50)

# Create summary table
summary_df = pd.DataFrame({
    'Algorithm': models,
    'R2_Score': [results[model]['r2'] for model in models],
    'RMSE': [results[model]['rmse'] for model in models],
    'MAE': [results[model]['mae'] for model in models],
    'CV_Mean': [cv_results[model].mean() for model in models],
    'CV_Std': [cv_results[model].std() for model in models]
}).sort_values('R2_Score', ascending=False)

print("\nFinal Rankings (by R² Score):")
print(summary_df.to_string(index=False, float_format='%.3f'))

# Best model analysis
best_algorithm = summary_df.iloc[0]['Algorithm']
best_r2 = summary_df.iloc[0]['R2_Score']
best_rmse = summary_df.iloc[0]['RMSE']

print(f"\nBEST PERFORMING ALGORITHM: {best_algorithm}")
print(f"Performance: R² = {best_r2:.3f}, RMSE = ${best_rmse:.0f}")
print(f"Model explains {best_r2:.1%} of price variation")

# Algorithm insights
print(f"\nALGORITHM INSIGHTS:")
print(f"1. Most accurate: {summary_df.iloc[0]['Algorithm']} (R² = {summary_df.iloc[0]['R2_Score']:.3f})")
print(f"2. Most stable: {summary_df.loc[summary_df['CV_Std'].idxmin(), 'Algorithm']} (CV std = {summary_df['CV_Std'].min():.3f})")
print(f"3. Performance range: {summary_df['R2_Score'].min():.3f} to {summary_df['R2_Score'].max():.3f}")

# Feature engineering impact
if best_tree_model:
    model = results[best_tree_model]['model']
    engineered_features = ['rooms_total', 'sqft_per_room']
    engineered_importance = sum(model.feature_importances_[feature_columns.index(f)] for f in engineered_features if f in feature_columns)
    print(f"4. Engineered features contribute {engineered_importance:.1%} to model decisions")

print(f"\nRECOMMENDATION:")
print(f"Use {best_algorithm} for this prediction task")
print(f"Expected prediction accuracy: ±${summary_df.iloc[0]['MAE']:.0f} on average")

print(f"\nProject completed successfully!")