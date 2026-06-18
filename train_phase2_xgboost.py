"""
Phase 2 - Step 2a: XGBoost 학습
- extract_features.py로 추출한 features.npz 사용
- 시계열 윈도우 특징 추가 (현재 + 과거 N프레임 mean + max)
- 영상 단위 train/val 분할 (같은 영상 프레임이 양쪽에 들어가지 않음)
- 영상 단위 평가 (한 프레임이라도 fall 예측되면 positive)
"""
import os
import time
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_PATH = Path(r"C:\fall-detection\phase2_features\features.npz")
MODEL_PATH = Path(r"C:\fall-detection\phase2_features\xgboost_model.json")

WINDOW = 8                # 과거 N프레임 통계
MIN_CONSECUTIVE = 3       # 영상 단위 평가에서 fall 확정에 필요한 연속 프레임
PRED_THRESHOLD = 0.5      # XGBoost 확률 → fall 판정 임계


def build_window_features(seq):
    """[T, F] → [T, F*3]: 현재 + 과거 N의 mean + max"""
    T, F = seq.shape
    out = np.zeros((T, F * 3), dtype=np.float32)
    for t in range(T):
        s = max(0, t - WINDOW + 1)
        w = seq[s:t+1]
        out[t, :F] = seq[t]
        out[t, F:2*F] = w.mean(axis=0)
        out[t, 2*F:3*F] = w.max(axis=0)
    return out


def main():
    print("=" * 70)
    print("Phase 2 - Step 2a: XGBoost 학습")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        print(f"[에러] {FEATURES_PATH} 없음. 먼저 'python extract_features.py' 실행하세요.")
        return

    data = np.load(FEATURES_PATH, allow_pickle=True)
    all_feats = data["feats"]
    all_labels = data["labels"]
    all_meta = data["meta"]

    n_videos = len(all_feats)
    print(f"  로드 완료: {n_videos}개 영상, 총 {sum(len(f) for f in all_feats)} 프레임")

    # 영상 단위 분할 (80/20)
    np.random.seed(42)
    indices = np.arange(n_videos)
    np.random.shuffle(indices)
    n_train = int(n_videos * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    print(f"  Train 영상: {len(train_idx)} / Val 영상: {len(val_idx)}")

    # 시계열 윈도우 특징 구축
    print(f"  시계열 윈도우 특징 구축 (WINDOW={WINDOW})")
    t0 = time.time()
    X_train_list, y_train_list = [], []
    for i in train_idx:
        f = build_window_features(all_feats[i])
        X_train_list.append(f)
        y_train_list.append(all_labels[i])
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list).astype(int)

    X_val_list, y_val_list, val_group_lens = [], [], []
    val_meta = []
    for i in val_idx:
        f = build_window_features(all_feats[i])
        X_val_list.append(f)
        y_val_list.append(all_labels[i])
        val_group_lens.append(len(f))
        val_meta.append(all_meta[i])
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list).astype(int)
    print(f"  특징 차원: {X_train.shape[1]} (현재 53 + mean 53 + max 53)")
    print(f"  Train 프레임: {len(X_train)} (fall: {(y_train==1).sum()})")
    print(f"  Val   프레임: {len(X_val)} (fall: {(y_val==1).sum()})")
    print(f"  구축 시간: {(time.time()-t0):.1f}초")

    # 클래스 불균형 보정
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  scale_pos_weight: {pos_w:.2f}")

    # 학습
    print("\n[XGBoost 학습 시작]")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_w,
        n_jobs=-1,
        eval_metric=["logloss", "error"],
        early_stopping_rounds=30,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"  학습 시간: {(time.time()-t0)/60:.1f}분")

    # 프레임 단위 평가
    print("\n[프레임 단위 평가]")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, y_pred, target_names=["no_fall", "fall"]))
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix:\n  TN={cm[0,0]}, FP={cm[0,1]}\n  FN={cm[1,0]}, TP={cm[1,1]}")

    # 영상 단위 평가 (실제 운영 평가 방식)
    print("\n[영상 단위 평가]")
    tp = fn = tn = fp = 0
    cursor = 0
    for i, glen in enumerate(val_group_lens):
        v_pred = y_pred[cursor:cursor+glen]
        v_true = y_val[cursor:cursor+glen]
        cursor += glen

        # 영상에 fall=1 프레임이 하나라도 있으면 진짜 낙상 영상
        is_fall_video = (v_true.sum() > 0)
        # 예측: MIN_CONSECUTIVE 연속 fall 프레임이 있으면 fall 영상으로 판정
        pred_fall_video = False
        consecutive = 0
        for p in v_pred:
            if p == 1:
                consecutive += 1
                if consecutive >= MIN_CONSECUTIVE:
                    pred_fall_video = True
                    break
            else:
                consecutive = 0

        if is_fall_video and pred_fall_video: tp += 1
        elif is_fall_video and not pred_fall_video: fn += 1
        elif not is_fall_video and pred_fall_video: fp += 1
        else: tn += 1

    total = tp + fn + tn + fp
    acc = (tp + tn) / total * 100 if total else 0
    sens = tp / (tp + fn) * 100 if (tp + fn) else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0

    print(f"  영상 수: {total}  (낙상영상 {tp+fn}, 정상영상 {tn+fp})")
    print(f"  TP:{tp}  FN:{fn}  TN:{tn}  FP:{fp}")
    print(f"  정확도:    {acc:.1f}%")
    print(f"  민감도:    {sens:.1f}%")
    print(f"  특이도:    {spec:.1f}%")
    print(f"  정밀도:    {prec:.1f}%")
    print(f"  F1 Score:  {f1:.1f}")

    # 특징 중요도 (상위 15)
    print("\n[특징 중요도 Top 15]")
    importances = model.feature_importances_
    feat_names = []
    base_names = (
        [f"kp{i}_{c}" for i in range(13) for c in ("x","y","v")]
        + ["horiz", "sh_diff", "hk_diff", "vel", "acc"]
        + ["pos_lying", "pos_sit", "pos_stand", "pos_unk"]
        + ["yolo_fall"]
        + ["mae_fall", "mae_lying", "mae_sit", "mae_walk"]
    )
    feat_names = (
        [f"cur_{n}" for n in base_names]
        + [f"mean_{n}" for n in base_names]
        + [f"max_{n}" for n in base_names]
    )
    top = np.argsort(importances)[::-1][:15]
    for idx in top:
        print(f"  {feat_names[idx]:30s} : {importances[idx]:.4f}")

    # 모델 저장
    model.save_model(str(MODEL_PATH))
    print(f"\n모델 저장: {MODEL_PATH}")
    print(f"다음 단계: python train_phase2_cnn.py  (1D-CNN 학습) 또는")
    print(f"          python train_phase2_lstm.py  (LSTM 학습)")


if __name__ == "__main__":
    main()
