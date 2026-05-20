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

def run(target):
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

    y = df.iloc[valid_indices][target].values
    X_descs = np.array(X_descs)

    X_imp   = SimpleImputer(strategy='mean').fit_transform(X_descs)
    X_var   = VarianceThreshold(threshold=0.01).fit_transform(X_imp)
    X_sc    = StandardScaler().fit_transform(X_var)
    X_final = remove_collinear_features(X_sc, threshold=0.95)

    models = [
        (LinearRegression(),                                                              "Linear Regression"),
        (DecisionTreeRegressor(max_depth=5, random_state=42),                            "Single Tree (d=5)"),
        (RandomForestRegressor(n_estimators=300, min_samples_leaf=4,
                               max_samples=0.5, random_state=42, n_jobs=-1),             "Random Forest"),
    ]

    print(f"\n{'─'*70}")
    print(f"  타겟: {target}   |   특성 수: {X_final.shape[1]}")
    print(f"{'─'*70}")
    print(f"  {'모델':<25} {'Train R²':>10} {'CV Test R²':>12} {'Gap':>8}")
    print(f"{'─'*70}")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for model, name in models:
        s = cross_validate(model, X_final, y, cv=cv, scoring='r2', return_train_score=True)
        tr, te = s['train_score'].mean(), s['test_score'].mean()
        print(f"  {name:<25} {tr:>10.4f} {te:>12.4f} {tr-te:>8.4f}")

# ============================================================
# 도전 1: 세 타겟의 예측 난이도를 비교해보세요.
# δD / δP / δH 순서로 실행하며 CV Test R² 차이를 관찰하고
# 왜 그런 차이가 나는지 화학적으로 설명해보세요.
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  도전 1: δD / δP / δH — 타겟별 예측 난이도 비교")
    print("=" * 70)
    for t in ["δD    (MPa0.5)", "δP    (MPa0.5)", "δH    (MPa0.5)"]:
        run(t)
    print()
