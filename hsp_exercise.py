import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

def remove_collinear_features(X, threshold=0.95):
    df_X = pd.DataFrame(X)
    corr_matrix = df_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    to_keep = [c for c in df_X.columns if c not in to_drop]
    return X[:, to_keep]

# ============================================================
# 🎯 실습 과제: 아래 TARGET 변수를 바꿔가며 결과를 비교해보세요.
#
#   "δD    (MPa0.5)"  →  분산력 (Dispersion)
#   "δP    (MPa0.5)"  →  극성   (Polarity)
#   "δH    (MPa0.5)"  →  수소결합력 (Hydrogen Bonding)  ← 예시로 제공된 타겟
#
# 각 타겟에 대해 세 모델의 R2 점수가 어떻게 달라지는지 관찰하고,
# 왜 차이가 나는지 화학적으로 설명해보세요.
# ============================================================
TARGET = "δH    (MPa0.5)"

def main():
    print("=" * 70)
    print(f"🎯 HSP 소규모 데이터셋: 왜 트리(Tree) 기반 모델인가?")
    print(f"   예측 타겟: {TARGET}")
    print("=" * 70)

    df = pd.read_csv('hsp.csv')
    mols = [Chem.MolFromSmiles(str(s)) if pd.notnull(s) else None for s in df['Smiles']]

    desc_names = [x[0] for x in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    X_descs, valid_indices = [], []
    for i, mol in enumerate(mols):
        if mol:
            try:
                X_descs.append(calc.CalcDescriptors(mol))
                valid_indices.append(i)
            except:
                pass

    y = df.iloc[valid_indices][TARGET].values
    X_descs = np.array(X_descs)

    # 전처리 및 프루닝
    X_descs_imp = SimpleImputer(strategy='mean').fit_transform(X_descs)
    X_descs_var = VarianceThreshold(threshold=0.01).fit_transform(X_descs_imp)
    X_descs_scaled = StandardScaler().fit_transform(X_descs_var)
    X_descs_pruned = remove_collinear_features(X_descs_scaled, threshold=0.95)

    print(f"✔️  사용된 최적화 특성(Descriptors Only): {X_descs_pruned.shape[1]} 개\n")

    def evaluate(model, name, desc):
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(model, X_descs_pruned, y, cv=cv, scoring='r2', return_train_score=True)
        train_r2 = scores['train_score'].mean()
        test_r2 = scores['test_score'].mean()
        gap = train_r2 - test_r2
        print(f"[{name}]")
        print(f"  설명: {desc}")
        print(f"  Train R2: {train_r2:.4f} | CV Test R2: {test_r2:.4f} (Gap: {gap:.4f})")
        print("-" * 70)

    models = [
        (LinearRegression(),
         "1. 선형 회귀 (Linear Regression)",
         "가장 단순한 모델. 변수가 많을 때 과적합이 심하게 발생함."),

        (DecisionTreeRegressor(max_depth=5, random_state=42),
         "2. 단일 의사결정 나무 (Single Tree, depth=5)",
         "물리화학적 분기 조건을 만들기 시작하지만 단일 트리의 한계가 있음."),

        (RandomForestRegressor(n_estimators=300, min_samples_leaf=4, max_samples=0.5, random_state=42, n_jobs=-1),
         "3. Random Forest (안정화 모델)",
         "⭐수백 개의 트리가 집단 지성을 발휘하여 과적합을 억제함.")
    ]

    for m, name, desc in models:
        evaluate(m, name, desc)

if __name__ == '__main__':
    main()
