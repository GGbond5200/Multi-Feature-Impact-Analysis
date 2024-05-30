#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from imblearn.over_sampling import SMOTE


df = pd.read_csv('E:/dataset/BugHunterDataset-1.0/full/all/class.csv')


# In[3]:


columns_to_remove = ['Project', 'Hash', 'LongName']
target = 'Number of Bugs'
all_features = df.drop(columns_to_remove + [target], axis=1)
all_features = all_features.replace([np.inf, -np.inf], np.nan)
all_features = all_features.dropna()
df = df[df.index.isin(all_features.index)]


# In[7]:


data = df


# In[4]:






# In[5]:


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


# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# 
# # 定义参数网格
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [None, 10, 20, 30],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# 
# # 初始化模型
# rf = RandomForestClassifier()
# 
# # 使用网格搜索进行超参数优化
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
# 
# # 输出最佳参数
# print("Best parameters found: ", grid_search.best_params_)
# 

# In[ ]:





# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# 去除非特征列
columns_to_remove = ['Project', 'Hash', 'LongName', 'Number of Bugs']
data_cleaned = data.drop(columns_to_remove, axis=1)

# 选择前20个特征进行可视化
selected_features = data_cleaned.columns[:20]
data_subset = data_cleaned[selected_features]

# 计算选定特征的相关系数矩阵
corr_matrix_subset = data_subset.corr()

# 初始化选定特征的p值矩阵
p_values_subset = np.zeros_like(corr_matrix_subset)

# 计算每对特征的p值
for i in range(corr_matrix_subset.shape[0]):
    for j in range(corr_matrix_subset.shape[1]):
        if i != j:
            _, p = pearsonr(data_subset.iloc[:, i], data_subset.iloc[:, j])
            p_values_subset[i, j] = p
        else:
            p_values_subset[i, j] = 0  # 设置对角线的p值为0

# 将p值矩阵转换为DataFrame
p_value_matrix_subset = pd.DataFrame(p_values_subset, columns=selected_features, index=selected_features)

# 设置字体样式为New Times Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 可视化选定特征的相关系数矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_subset, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix (First 20 Features)')
plt.show()

# 可视化选定特征的p值矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(p_value_matrix_subset, annot=True, fmt=".4f", cmap='coolwarm', vmax=0.05)
plt.title('P-Value Matrix (First 20 Features)')
plt.show()


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


# 去除非特征列
columns_to_remove = ['Project', 'Hash', 'LongName', 'Number of Bugs']
data_cleaned = data.drop(columns_to_remove, axis=1)

# 选择前20个特征进行可视化
selected_features = data_cleaned.columns[:20]
data_subset = data_cleaned[selected_features]

# 计算选定特征的相关系数矩阵
corr_matrix_subset = data_subset.corr()

# 初始化选定特征的p值矩阵
p_values_subset = np.zeros_like(corr_matrix_subset)

# 计算每对特征的p值
for i in range(corr_matrix_subset.shape[0]):
    for j in range(corr_matrix_subset.shape[1]):
        if i != j:
            _, p = pearsonr(data_subset.iloc[:, i], data_subset.iloc[:, j])
            p_values_subset[i, j] = p
        else:
            p_values_subset[i, j] = 0  # 设置对角线的p值为0

# 将p值矩阵转换为DataFrame
p_value_matrix_subset = pd.DataFrame(p_values_subset, columns=selected_features, index=selected_features)

# 初始化比较结果的DataFrame
comparison_results = pd.DataFrame(index=selected_features, columns=selected_features)

# 比较相关系数矩阵和p值矩阵的每个元素
for i in range(corr_matrix_subset.shape[0]):
    for j in range(corr_matrix_subset.shape[1]):
        corr_value = corr_matrix_subset.iloc[i, j]
        p_value = p_value_matrix_subset.iloc[i, j]
        
        # 判断结果是否一致
        if (abs(corr_value) >= 0.5 and p_value < 0.05) or (abs(corr_value) < 0.5 and p_value >= 0.05):
            comparison_results.iloc[i, j] = "Consistent"
        else:
            comparison_results.iloc[i, j] = "Inconsistent"

# 显示比较结果
comparison_results


# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np


# 选择前20个特征进行可视化
selected_features = data_cleaned.columns[:20]
data_subset = data_cleaned[selected_features]

# 计算选定特征的Spearman相关系数矩阵
spearman_corr_matrix = data_subset.corr(method='spearman')

# 初始化选定特征的Spearman p值矩阵
spearman_p_values = np.zeros_like(spearman_corr_matrix)

# 计算每对特征的Spearman相关系数和p值
for i in range(spearman_corr_matrix.shape[0]):
    for j in range(spearman_corr_matrix.shape[1]):
        if i != j:
            _, p = spearmanr(data_subset.iloc[:, i], data_subset.iloc[:, j])
            spearman_p_values[i, j] = p
        else:
            spearman_p_values[i, j] = 0  # 设置对角线的p值为0

# 将Spearman p值矩阵转换为DataFrame
spearman_p_value_matrix = pd.DataFrame(spearman_p_values, columns=selected_features, index=selected_features)

# 设置字体样式为New Times Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 可视化选定特征的Spearman相关系数矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Spearman Correlation Matrix (First 20 Features)')
plt.show()

# 可视化选定特征的Spearman p值矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_p_value_matrix, annot=True, fmt=".4f", cmap='coolwarm', vmax=0.05)
plt.title('Spearman P-Value Matrix (First 20 Features)')
plt.show()

# 初始化比较结果的DataFrame
spearman_comparison_results = pd.DataFrame(index=selected_features, columns=selected_features)

# 比较Spearman相关系数矩阵和Spearman p值矩阵的每个元素
for i in range(spearman_corr_matrix.shape[0]):
    for j in range(spearman_corr_matrix.shape[1]):
        corr_value = spearman_corr_matrix.iloc[i, j]
        p_value = spearman_p_value_matrix.iloc[i, j]
        
        # 判断结果是否一致
        if (abs(corr_value) >= 0.5 and p_value < 0.05) or (abs(corr_value) < 0.5 and p_value >= 0.05):
            spearman_comparison_results.iloc[i, j] = "Consistent"
        else:
            spearman_comparison_results.iloc[i, j] = "Inconsistent"

# 显示比较结果
spearman_comparison_results


# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np



# 计算Spearman相关系数矩阵
spearman_corr_matrix = data_cleaned.corr(method='spearman')

# 初始化Spearman p值矩阵
spearman_p_values = np.zeros_like(spearman_corr_matrix)

# 计算每对特征的Spearman相关系数和p值
for i in range(spearman_corr_matrix.shape[0]):
    for j in range(spearman_corr_matrix.shape[1]):
        if i != j:
            _, p = spearmanr(data_cleaned.iloc[:, i], data_cleaned.iloc[:, j])
            spearman_p_values[i, j] = p
        else:
            spearman_p_values[i, j] = 0  # 设置对角线的p值为0

# 将Spearman p值矩阵转换为DataFrame
spearman_p_value_matrix = pd.DataFrame(spearman_p_values, columns=data_cleaned.columns, index=data_cleaned.columns)

# 提取关联度最强的30对特征（除了自己与自己的关联）
abs_corr_matrix = spearman_corr_matrix.abs()
np.fill_diagonal(abs_corr_matrix.values, 0)  # 将对角线上的值设为0，以排除自己与自己的关联
strongest_pairs = abs_corr_matrix.unstack().sort_values(ascending=False).head(30).index

# 检查这些最强关联的p值是否显著
strong_corrs = []
for pair in strongest_pairs:
    feature1, feature2 = pair
    corr_value = spearman_corr_matrix.loc[feature1, feature2]
    p_value = spearman_p_value_matrix.loc[feature1, feature2]
    strong_corrs.append((feature1, feature2, corr_value, p_value, 'Significant' if p_value < 0.05 else 'Not Significant'))

# 将结果转换为DataFrame以便查看
strong_corrs_df = pd.DataFrame(strong_corrs, columns=['Feature1', 'Feature2', 'Spearman Correlation', 'P-Value', 'Significance'])

# 显示结果
print(strong_corrs_df)

# 可视化选定特征的Spearman相关系数矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Spearman Correlation Matrix')
plt.show()

# 可视化选定特征的Spearman p值矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_p_value_matrix, annot=True, fmt=".4f", cmap='coolwarm', vmax=0.05)
plt.title('Spearman P-Value Matrix')
plt.show()



# In[13]:


from pdpbox import pdp


# In[18]:


# 生成PDP交互图
pdp_interact_out = pdp.pdp_interact(
    model=class_models['random_forest']['fitted'],
    dataset=pd.concat((X_test, y_test), axis=1),
    model_features=X_test.columns.tolist(),
    features=['CLC', 'CC'],
    n_jobs=-1
)

# 绘制PDP交互图
fig, axes = pdp.pdp_interact_plot(pdp_interact_out, ['CLC', 'CC'], plot_type='contour')

# 保存图像
plt.savefig('C:/Users/10956/Desktop/code/666.png')
plt.show()


# In[15]:


import hashlib
import matplotlib.pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_plot(model, X, features):
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

    ax = axes['pdp_inter_ax']
    for text in ax.texts:
        text.set_fontsize(18)
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=18)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)

    if 'cbar_ax' in axes:
        cbar_ax = axes['cbar_ax']
        cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=18)
        cbar = cbar_ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.savefig('plot_output.png')
    plt.close(fig)
    return 'plot_output.png'

def calculate_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 数据模拟和模型训练
data = pd.DataFrame({
    'total operators': np.random.randint(1, 10, 100),
    'total operands': np.random.randint(1, 10, 100),
    'target': np.random.randint(0, 2, 100)
})
X = data[['total operators', 'total operands']]
y = data['target']

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 多次生成图像并计算哈希值
num_iterations = 20
hashes = []

for _ in range(num_iterations):
    filename = generate_plot(model, X, ['total operators', 'total operands'])
    hash_value = calculate_md5(filename)
    hashes.append(hash_value)

# 计算一致性百分比
unique_hashes = set(hashes)
consistency_percentage = (1 - len(unique_hashes) / num_iterations) * 100

print(f"Consistency Percentage: {consistency_percentage}%")


# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


model = RandomForestClassifier().fit(X_train, y_train)
sample = X_test.iloc[0].copy()  

# 原始预测
original_pred = model.predict_proba(sample.values.reshape(1, -1))

# 修改特征值
sample['CLC'] += 0.1
sample['CC'] += 0.1

# 修改后的预测
modified_pred = model.predict_proba(sample.values.reshape(1, -1))

print("原始预测:", original_pred)
print("修改后预测:", modified_pred)


# In[26]:


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
 


# In[25]:


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
    modified_sample['CC'] += 0
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
 


# In[20]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X_train.columns, filled=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # question 2

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




