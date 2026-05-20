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

# ============================================================
# 도전 4: 커스텀 Descriptor로 가설 검증
# "δH는 -OH 개수에 비례한다"
#
# 핵심: num_OH와 상관이 높은 기존 Descriptor들을 먼저 제거해야
# num_OH의 기여도를 순수하게 볼 수 있습니다.
#
# 참고: 아래 변수들이 num_OH와 높은 상관을 보입니다.
#   fr_Al_OH        (r=0.97)  ← 지방족 OH 비율 — num_OH와 거의 동일
#   VSA_EState3     (r=0.94)  ← OH 관련 전자적 표면적
#   fr_Al_OH_noTert (r=0.93)  ← 3차 탄소 제외 지방족 OH
#   NHOHCount       (r=0.87)  ← N-H + O-H 총 개수
#   NumHDonors      (r=0.79)  ← 수소결합 공여자 수
# ============================================================
TARGET      = "δH    (MPa0.5)"
CUSTOM_NAME = "num_OH"
REMOVE_BEFORE = [
    'fr_Al_OH', 'VSA_EState3', 'fr_Al_OH_noTert',
    'NHOHCount', 'NumHDonors',
]

def num_OH(mol):
    """분자 내 -OH 작용기 개수"""
    return sum(1 for atom in mol.GetAtoms()
               if atom.GetAtomicNum() == 8
               and atom.GetTotalNumHs() >= 1
               and not atom.IsInRing())

if __name__ == '__main__':
    print("=" * 70)
    print(f"  도전 4: 커스텀 Descriptor [{CUSTOM_NAME}] 가설 검증")
    print(f"  사전 제거 변수 ({len(REMOVE_BEFORE)}개): {REMOVE_BEFORE}")
    print("=" * 70)

    df       = pd.read_csv('hsp.csv')
    mols_raw = [Chem.MolFromSmiles(str(s)) if pd.notnull(s) else None for s in df['Smiles']]

    desc_names = [x[0] for x in Descriptors.descList]
    calc       = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    X_raw, oh_vals, valid_indices = [], [], []
    for i, mol in enumerate(mols_raw):
        if mol:
            try:
                X_raw.append(calc.CalcDescriptors(mol))
                oh_vals.append(num_OH(mol))
                valid_indices.append(i)
            except:
                pass

    y       = df.iloc[valid_indices][TARGET].values
    X_raw   = np.array(X_raw)
    oh_vals = np.array(oh_vals)

    # ── 이름 추적하며 전처리
    X_imp  = SimpleImputer(strategy='mean').fit_transform(X_raw)
    vt     = VarianceThreshold(threshold=0.01)
    X_var  = vt.fit_transform(X_imp)
    names  = [desc_names[i] for i in vt.get_support(indices=True)]

    # ── 상관 변수 명시적 제거
    keep_mask      = [n not in REMOVE_BEFORE for n in names]
    X_filtered     = X_var[:, keep_mask]
    names_filtered = [n for n, k in zip(names, keep_mask) if k]
    removed        = [n for n in REMOVE_BEFORE if n in names]
    print(f"\n  실제 제거된 변수: {removed}")

    # ── num_OH 추가
    oh_col     = oh_vals.reshape(-1, 1).astype(float)
    X_combined = np.hstack([X_filtered, oh_col])
    names_all  = names_filtered + [CUSTOM_NAME]

    # ── 스케일링 + 다중공선성 제거
    X_sc     = StandardScaler().fit_transform(X_combined)
    df_corr  = pd.DataFrame(X_sc, columns=names_all)
    corr     = df_corr.corr().abs()
    upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop  = [c for c in upper.columns if any(upper[c] > 0.95)]
    to_keep  = [n for n in names_all if n not in to_drop]
    to_keep_idx = [names_all.index(n) for n in to_keep]
    X_final  = X_sc[:, to_keep_idx]

    survived = CUSTOM_NAME in to_keep
    print(f"  {CUSTOM_NAME} 생존 여부: {'✅ 살아남음' if survived else '❌ 프루닝에서 제거됨'}")
    print(f"  최종 특성 수: {X_final.shape[1]}개\n")

    # ── 기준선: num_OH 없이 (REMOVE_BEFORE만 빼고)
    X_base     = X_sc[:, [to_keep.index(n) for n in to_keep if n != CUSTOM_NAME]]
    names_base = [n for n in to_keep if n != CUSTOM_NAME]

    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=4,
                                  max_samples=0.5, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    s_base = cross_validate(model, X_base,  y, cv=cv, scoring='r2', return_train_score=True)
    s_oh   = cross_validate(model, X_final, y, cv=cv, scoring='r2', return_train_score=True)

    tr_b, te_b = s_base['train_score'].mean(), s_base['test_score'].mean()
    tr_o, te_o = s_oh['train_score'].mean(),   s_oh['test_score'].mean()

    print(f"  {'모델':<35} {'특성 수':>6}  {'Train R²':>9}  {'CV Test R²':>10}  {'Gap':>7}")
    print(f"  {'─'*70}")
    print(f"  {'상관 변수 제거 (num_OH 없음)':<35} {X_base.shape[1]:>6}  {tr_b:>9.4f}  {te_b:>10.4f}  {tr_b-te_b:>7.4f}")
    if survived:
        print(f"  {'+ num_OH 추가':<35} {X_final.shape[1]:>6}  {tr_o:>9.4f}  {te_o:>10.4f}  {tr_o-te_o:>7.4f}")
        delta = te_o - te_b
        print(f"\n  num_OH 추가로 인한 CV R² 변화: {delta:+.4f}")

    # ── Feature Importance
    model.fit(X_final, y)
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    print(f"\n  Feature Importance 상위 10개:")
    print(f"  {'순위':<5} {'Descriptor':<35} {'Importance':>10}")
    print(f"  {'─'*55}")
    for rank, idx in enumerate(top_idx, 1):
        name   = to_keep[idx]
        marker = " ★" if name == CUSTOM_NAME else ""
        print(f"  {rank:<5} {name+marker:<35} {importances[idx]:>10.4f}")
    print()
