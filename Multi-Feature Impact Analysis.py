import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from imblearn.over_sampling import SMOTE


df = pd.read_csv('path ')


# Data processing

columns_to_remove = ['Project', 'Hash', 'LongName']
target = 'Number of Bugs'
all_features = df.drop(columns_to_remove + [target], axis=1)
all_features = all_features.replace([np.inf, -np.inf], np.nan)
all_features = all_features.dropna()
df = df[df.index.isin(all_features.index)]

data = df[df.index.isin(all_features.index)]
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])

X = data.drop(columns_to_remove + [target], axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# model training

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, linear_model, tree, discriminant_analysis, ensemble, neural_network
from sklearn.metrics import matthews_corrcoef
import random  # Importing the random module

# Generate a random integer, range can be adjusted as needed
rand = random.randint(1, 100)

param_grid = {
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': rand,
    'class_weight': 'balanced'
}

class_models = {
    'decision_tree': {'model': make_pipeline(tree.DecisionTreeClassifier(**param_grid))},
    'gradient_boosting': {'model': make_pipeline(ensemble.GradientBoostingClassifier(n_estimators=250, max_depth=6, subsample=0.8, learning_rate=0.02))},
    'random_forest': {'model': make_pipeline(ensemble.RandomForestClassifier(max_depth=3, n_estimators=200, max_features='sqrt', random_state=rand))},
    'logistic': {'model': make_pipeline(linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs', class_weight='balanced', max_iter=100))},
    'lda': {'model': make_pipeline(discriminant_analysis.LinearDiscriminantAnalysis(n_components=1))},
    'mlp': {'model': make_pipeline(StandardScaler(), neural_network.MLPClassifier(hidden_layer_sizes=(11,20), early_stopping=True, random_state=rand, validation_fraction=0.25, max_iter=500))}
}

# Train and evaluate each model
for model_name in class_models.keys():
    fitted_model = class_models[model_name]['model'].fit(X_train, y_train)
    y_train_pred = fitted_model.predict(X_train)
    y_test_pred = fitted_model.predict(X_test)
    class_models[model_name]['fitted'] = fitted_model
    class_models[model_name]['preds'] = y_test_pred
    class_models[model_name]['Accuracy_train'] = metrics.accuracy_score(y_train, y_train_pred)
    class_models[model_name]['Accuracy_test'] = metrics.accuracy_score(y_test, y_test_pred)
    class_models[model_name]['Recall_train'] = metrics.recall_score(y_train, y_train_pred, average='weighted')
    class_models[model_name]['Recall_test'] = metrics.recall_score(y_test, y_test_pred, average='weighted')
    class_models[model_name]['Precision_train'] = metrics.precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
    class_models[model_name]['Precision_test'] = metrics.precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
    class_models[model_name]['F1_test'] = metrics.f1_score(y_test, y_test_pred, average='weighted')
    class_models[model_name]['MCC_test'] = matthews_corrcoef(y_test, y_test_pred)

import pandas as pd
import numpy as np

# Create a DataFrame containing performance metrics of different models
class_metrics = pd.DataFrame.from_dict(class_models, 'index')[['Accuracy_train', 'Accuracy_test', 'Recall_train', 'Recall_test', 'Precision_train', 'Precision_test', 'F1_test', 'MCC_test']]
with pd.option_context('display.precision', 3):
    html = class_metrics.sort_values(by='MCC_test', ascending=False).style.background_gradient(cmap='plasma', low=0.43, high=0.63, subset=['Accuracy_train', 'Accuracy_test']).background_gradient(cmap='viridis', low=0.63, high=0.43, subset=['F1_test'])

display(html)
print('NIR: %.4f' % (y_train[y_train == 1].shape[0] / y_train.shape[0]))



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

selected_features = data_cleaned.columns[:20]
data_subset = data_cleaned[selected_features]

# Spearman matrix
spearman_corr_matrix = data_subset.corr(method='spearman')

#  p-value
spearman_p_values = np.zeros_like(spearman_corr_matrix)
for i in range(spearman_corr_matrix.shape[0]):
    for j in range(spearman_corr_matrix.shape[1]):
        if i != j:
            _, p = spearmanr(data_subset.iloc[:, i], data_subset.iloc[:, j])
            spearman_p_values[i, j] = p
        else:
            spearman_p_values[i, j] = 0  
spearman_p_value_matrix = pd.DataFrame(spearman_p_values, columns=selected_features, index=selected_features)


plt.rcParams['font.family'] = 'Times New Roman'

# Spearman Correlation coefficient matrix
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Spearman Correlation Matrix (First 20 Features)')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_p_value_matrix, annot=True, fmt=".4f", cmap='coolwarm', vmax=0.05)
plt.title('Spearman P-Value Matrix (First 20 Features)')
plt.show()
spearman_comparison_results = pd.DataFrame(index=selected_features, columns=selected_features)
for i in range(spearman_corr_matrix.shape[0]):
    for j in range(spearman_corr_matrix.shape[1]):
        corr_value = spearman_corr_matrix.iloc[i, j]
        p_value = spearman_p_value_matrix.iloc[i, j]

        if (abs(corr_value) >= 0.5 and p_value < 0.05) or (abs(corr_value) < 0.5 and p_value >= 0.05):
            spearman_comparison_results.iloc[i, j] = "Consistent"
        else:
            spearman_comparison_results.iloc[i, j] = "Inconsistent"
spearman_comparison_results


#Displays the ranking of the correlation coefficients
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

spearman_corr_matrix = data_cleaned.corr(method='spearman')
spearman_p_values = np.zeros_like(spearman_corr_matrix)
for i in range(spearman_corr_matrix.shape[0]):
    for j in range(spearman_corr_matrix.shape[1]):
        if i != j:
            _, p = spearmanr(data_cleaned.iloc[:, i], data_cleaned.iloc[:, j])
            spearman_p_values[i, j] = p
        else:
            spearman_p_values[i, j] = 0 

spearman_p_value_matrix = pd.DataFrame(spearman_p_values, columns=data_cleaned.columns, index=data_cleaned.columns)

abs_corr_matrix = spearman_corr_matrix.abs()
np.fill_diagonal(abs_corr_matrix.values, 0)  
strongest_pairs = abs_corr_matrix.unstack().sort_values(ascending=False).head(30).index

strong_corrs = []
for pair in strongest_pairs:
    feature1, feature2 = pair
    corr_value = spearman_corr_matrix.loc[feature1, feature2]
    p_value = spearman_p_value_matrix.loc[feature1, feature2]
    strong_corrs.append((feature1, feature2, corr_value, p_value, 'Significant' if p_value < 0.05 else 'Not Significant'))

strong_corrs_df = pd.DataFrame(strong_corrs, columns=['Feature1', 'Feature2', 'Spearman Correlation', 'P-Value', 'Significance'])

print(strong_corrs_df)

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Spearman Correlation Matrix')
plt.show()
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_p_value_matrix, annot=True, fmt=".4f", cmap='coolwarm', vmax=0.05)
plt.title('Spearman P-Value Matrix')
plt.show()




# Display pdp plot
from pdpbox import pdp
pdp_interact_out = pdp.pdp_interact(
    model=class_models['random_forest']['fitted'],
    dataset=pd.concat((X_test, y_test), axis=1),
    model_features=X_test.columns.tolist(),
    features=['CLC', 'CC'],
    n_jobs=-1
)
fig, axes = pdp.pdp_interact_plot(pdp_interact_out, ['CLC', 'CC'], plot_type='contour')
plt.show()



#Displaying iteration results
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pdpbox import pdp
learning_rates = [0.01, 0.02, 0.05, 0.1, 0.2]
results = []
for lr in learning_rates:

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_features=lr)
    model.fit(X_train, y_train)
    
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=X_test,
        model_features=X_test.columns.tolist(),
        features=['CLC', 'CC'],
        n_jobs=-1
    )  
    fig, axes = pdp.pdp_interact_plot(pdp_interact_out, ['CLC', 'CC'], plot_type='contour')
    plt.suptitle(f'PDP Interact for Learning Rate: {lr}')
    plt.show()

    results.append((lr, pdp_interact_out))

for lr, pdp_interact_out in results:
    print(f"Learning Rate: {lr}")
    display(pdp_interact_out)



# Impact of Learning Rate on PDP Values
import numpy as np
import matplotlib.pyplot as plt
from pdpbox import pdp


pdp_values = []

for lr, pdp_interact_out in results:
    
    pdp_value = pdp_interact_out[0].pdp['preds']
    pdp_values.append(pdp_value)

baseline_pdp_value = pdp_values[0]


mse_values = []
for pdp_value in pdp_values:
    mse = np.mean((pdp_value - baseline_pdp_value) ** 2)
    mse_values.append(mse)

learning_rates = [0.01, 0.02, 0.05, 0.1, 0.2]

plt.figure(figsize=(10, 6))
plt.plot(learning_rates, mse_values, marker='o', linewidth=3, label='MSE of PDP Values')  # 将linewidth增加到3
plt.xlabel('Learning Rate', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.title('Impact of Learning Rate on PDP Values', fontsize=16)
plt.legend(fontsize=14)  
plt.grid(True)
plt.xticks(learning_rates, fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('learning_rate_mse_plot.png')
plt.show()














import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_plot(model, X, features):
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=X,
        model_features=X.columns,
        features=features
    )
    pdp_values = [out.pdp['preds'] for out in pdp_interact_out]
    return pdp_values

columns_to_remove = ['Project', 'Hash', 'LongName']
target = 'Number of Bugs'

all_features = df.drop(columns_to_remove + [target], axis=1)
all_features = all_features.replace([np.inf, -np.inf], np.nan)
all_features = all_features.dropna()

data = df[df.index.isin(all_features.index)]
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])

X = data.drop(columns_to_remove + [target], axis=1)
y = data[target]

n_estimators_list = [10, 50, 100]
max_depth_list = [5, 10, 20]
min_samples_split_list = [2, 5, 10]

results = []

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                max_features='auto'
            )
            model.fit(X, y)
            
            pdp_values = generate_plot(model, X, ['CLC', 'CC'])
            baseline_pdp_value = pdp_values[0]
            mse = np.mean([(pdp_value - baseline_pdp_value) ** 2 for pdp_value in pdp_values])
            
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'mse': mse
            })

results_df = pd.DataFrame(results)

# Plotting n_estimators
fig, ax = plt.subplots(figsize=(15, 8))
for max_depth in results_df['max_depth'].unique():
    subset = results_df[results_df['max_depth'] == max_depth]
    ax.plot(subset['n_estimators'], subset['mse'], label=f'max_depth={max_depth}', marker='o')

ax.set_xlabel('n_estimators', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
ax.set_title('Impact of n_estimators on PDP Values', fontsize=20)
ax.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('n_estimators_impact.png')
plt.show()

# Plotting max_depth
fig, ax = plt.subplots(figsize=(15, 8))
for n_estimators in results_df['n_estimators'].unique():
    subset = results_df[results_df['n_estimators'] == n_estimators]
    ax.plot(subset['max_depth'], subset['mse'], label=f'n_estimators={n_estimators}', marker='o')

ax.set_xlabel('max_depth', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
ax.set_title('Impact of max_depth on PDP Values', fontsize=20)
ax.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('max_depth_impact.png')
plt.show()

# Plotting min_samples_split
fig, ax = plt.subplots(figsize=(15, 8))
for n_estimators in results_df['n_estimators'].unique():
    for max_depth in results_df['max_depth'].unique():
        subset = results_df[(results_df['n_estimators'] == n_estimators) & (results_df['max_depth'] == max_depth)]
        ax.plot(subset['min_samples_split'], subset['mse'], label=f'n={n_estimators}, d={max_depth}', marker='o')

ax.set_xlabel('min_samples_split', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
ax.set_title('Impact of min_samples_split on PDP Values', fontsize=20)
ax.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()








# Percentage Over Iterations
import hashlib
import matplotlib.pyplot as plt
from pdpbox import pdp
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generate_plot(model, X, features, iteration):
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=X,
        model_features=X.columns,
        features=features
    )
    fig, axes = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=features,
        plot_type='grid',
        x_quantile=True,
        ncols=2,
        figsize=(15, 15)
    )

    for ax in axes['pdp_inter_ax']:
        for text in ax.texts:
            text.set_fontsize(25)
        ax.tick_params(axis='both', labelsize=25)
        ax.set_xlabel(ax.get_xlabel(), fontsize=25)
        ax.set_ylabel(ax.get_ylabel(), fontsize=25)

    if 'cbar_ax' in axes:
        cbar_ax = axes['cbar_ax']
        cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=25)
        cbar = cbar_ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)

    plt.rcParams['font.family'] = 'Times New Roman'
    filename = f'plot_output_{iteration}.png'
    plt.savefig(filename)
    plt.close(fig)
    return filename

def calculate_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

df = df.head(500)
columns_to_remove = ['Project', 'Hash', 'LongName']
target = 'Number of Bugs'

all_features = df.drop(columns_to_remove + [target], axis=1)
all_features = all_features.replace([np.inf, -np.inf], np.nan)
all_features = all_features.dropna()


data = df[df.index.isin(all_features.index)]
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])

X = data.drop(columns_to_remove + [target], axis=1)
y = data[target]

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)


num_iterations = 40
hashes = []
consistency_data = []

for i in range(1, num_iterations + 1):
    filename = generate_plot(model, X, ['CLC', 'CC'], i)  # 替换 'CLC' 和 'CC' 为你的实际特征名称
    hash_value = calculate_md5(filename)
    hashes.append(hash_value)

    if i % 5 == 0:
        unique_hashes = set(hashes)
        consistency_percentage = (1 - len(unique_hashes) / i) * 100
        consistency_data.append((i, consistency_percentage))


consistency_df = pd.DataFrame(consistency_data, columns=['Iterations', 'Consistency Percentage'])

plt.figure(figsize=(10, 6))
plt.plot(consistency_df['Iterations'], consistency_df['Consistency Percentage'], marker='o', linewidth=3)  # 增加linewidth参数值
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('Consistency Percentage', fontsize=16)
plt.title('Consistency Percentage Over Iterations', fontsize=16)
plt.grid(True)
plt.xticks(consistency_df['Iterations'], fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('consistency_plot.png')
plt.show()





import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



# Initialize models
models = {
    "lda": LinearDiscriminantAnalysis(),
    "logistic": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "mlp": MLPClassifier()
}

# Function to modify sample
def modify_sample(sample):
    modified_sample = sample.copy()
    modified_sample['CLC'] += 0.3
    modified_sample['CC'] += 0.3
    return modified_sample

# Evaluate models
results = {}
for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Original prediction
    sample = X_test.iloc[0]
    original_pred = model.predict_proba(sample.values.reshape(1, -1))
    
    # Modify the sample
    modified_sample = modify_sample(sample)
    
    # Modified prediction
    modified_pred = model.predict_proba(modified_sample.values.reshape(1, -1))
    
    # Store results
    results[name] = {
        "original_pred": original_pred,
        "modified_pred": modified_pred
    }

# Print results
for name, result in results.items():
    print(f"{name.upper()} - Original Prediction: {result['original_pred']}, Modified Prediction: {result['modified_pred']}")
 


