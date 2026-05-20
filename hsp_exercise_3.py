import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

def remove_collinear_features(X, threshold=0.95):
    df_X = pd.DataFrame(X)
    corr_matrix = df_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    to_keep = [c for c in df_X.columns if c not in to_drop]
    return X[:, to_keep]

# ============================================================
# 도전 3: δH — Fingerprint를 추가하면 얼마나 올라가나?
# Descriptor만으로는 작용기의 '종류'를 구분 못합니다.
# Morgan Fingerprint를 더하면 그 한계가 어느 정도 해소되는지 확인하세요.
# ============================================================
TARGET = "δH    (MPa0.5)"

def run(X_final, y, label):
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=4,
                                  max_samples=0.5, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    s  = cross_validate(model, X_final, y, cv=cv, scoring='r2', return_train_score=True)
    tr, te = s['train_score'].mean(), s['test_score'].mean()
    print(f"  {label:<35} 특성 수: {X_final.shape[1]:>5}   Train: {tr:.4f}   CV Test: {te:.4f}   Gap: {tr-te:.4f}")

if __name__ == '__main__':
    print("=" * 70)
    print("  도전 3: δH — Descriptor만 vs Descriptor + Fingerprint")
    print("=" * 70)

    df   = pd.read_csv('hsp.csv')
    mols = [Chem.MolFromSmiles(str(s)) if pd.notnull(s) else None for s in df['Smiles']]

    desc_names = [x[0] for x in Descriptors.descList]
    calc       = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    fp_gen     = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    X_descs, X_fps, valid_indices = [], [], []
    for i, mol in enumerate(mols):
        if mol:
            try:
                X_descs.append(calc.CalcDescriptors(mol))
                X_fps.append(fp_gen.GetCountFingerprintAsNumPy(mol))
                valid_indices.append(i)
            except:
                pass

    y       = df.iloc[valid_indices][TARGET].values
    X_descs = np.array(X_descs)
    X_fps   = np.array(X_fps)

    # ── 공통 전처리
    X_imp    = SimpleImputer(strategy='mean').fit_transform(X_descs)
    X_var    = VarianceThreshold(threshold=0.01).fit_transform(X_imp)
    X_sc     = StandardScaler().fit_transform(X_var)
    X_pruned = remove_collinear_features(X_sc, threshold=0.95)

    # ── Descriptor만
    X_desc_only = X_pruned

    # ── Descriptor + Fingerprint
    X_with_fp = np.hstack([X_fps, X_pruned])

    print()
    run(X_desc_only, y, "Descriptor만 (프루닝 후)")
    run(X_with_fp,   y, "Descriptor + Morgan Fingerprint")
    print()
