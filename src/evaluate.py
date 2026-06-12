"""
Единая оценка трёх моделей на одинаковом TEST-сэмпле:
  1. Popularity baseline
  2. UserTopReorder baseline
  3. LightFM (гибрид)

Метрики (full-basket, включая reorder, см. главу 1.4 ВКР):
  - Precision@K — доля релевантных в топ-K рекомендаций
  - Recall@K    — доля найденных от всех релевантных
  - NDCG@K      — качество ранжирования (учёт позиций)

Для скорости оценка идёт на сэмпле из EVAL_USERS_SAMPLE юзеров,
имеющих непустой test-вектор (ground truth).

Графики сохраняются в figures/.
"""
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import joblib
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    TOP_K, RANDOM_SEED,
)
from src.logger import get_logger
# Импорт классов baseline-моделей нужен для joblib.load() —
# pickle ищет классы в неймспейсе, где их объявили
from src.baselines import PopularityRecommender, UserTopReorderRecommender  # noqa: F401
from src.hybrid import HybridRecommender

log = get_logger("evaluate", "evaluation.log")

# Сэмпл юзеров для оценки (полные 32к юзеров × 50к товаров в 1 поток LightFM = долго)
EVAL_USERS_SAMPLE = 5000


# === Универсальный интерфейс предсказания ===

class LightFMRecommender:
    """Обёртка над LightFM для унификации интерфейса с baseline-моделями."""

    def __init__(self, model, item_features):
        self.model = model
        self.item_features = item_features
        self.n_items = item_features.shape[0]
        self._all_items = np.arange(self.n_items)

    def recommend(self, user_idx: int, k: int = TOP_K, exclude_seen=None):
        scores = self.model.predict(
            user_ids=int(user_idx),
            item_ids=self._all_items,
            item_features=self.item_features,
        )
        if exclude_seen is not None:
            seen = exclude_seen[user_idx].indices
            scores[seen] = -np.inf  # не исключаем для full-basket, опция оставлена
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])]


# === Метрики ===

def precision_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / k


def recall_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """NDCG с бинарной релевантностью (см. главу 1.4.2 ВКР)."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(recommended[:k]):
        if r in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 потому что log2(1)=0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg

def hit_rate_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Hit Rate@K — 1, если хотя бы одна релевантная попала в топ-K, иначе 0.
    Бизнес-метрика: 'у скольких % юзеров мы угадали хоть один продукт'."""
    if not relevant:
        return 0.0
    return 1.0 if any(r in relevant for r in recommended[:k]) else 0.0


def evaluate_model(name: str, recommender, test: csr_matrix,
                   train: csr_matrix, eval_users: np.ndarray, k: int):
    log.info(f"Оценка модели: {name} на {len(eval_users)} юзерах")
    t0 = time.time()
    precisions, recalls, ndcgs, hits = [], [], [], []
    for user_idx in eval_users:
        relevant = set(test[user_idx].indices.tolist())
        if not relevant:
            continue
        recs = recommender.recommend(user_idx, k=k)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        hits.append(hit_rate_at_k(recs, relevant, k))
    elapsed = time.time() - t0
    result = {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "hit_rate_at_k": float(np.mean(hits)),
        "n_users_evaluated": len(precisions),
        "time_sec": elapsed,
    }
    log.info(f"  {name}: P@{k}={result['precision_at_k']:.4f} "
             f"R@{k}={result['recall_at_k']:.4f} "
             f"NDCG@{k}={result['ndcg_at_k']:.4f} "
             f"HitRate@{k}={result['hit_rate_at_k']:.4f} "
             f"(time {elapsed:.1f}s)")
    return result


# === Графики ===

def plot_metrics_comparison(results: dict, k: int):
    """Bar chart: сравнение моделей по трём метрикам."""
    models = list(results.keys())
    metrics = ["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]
    metric_labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", f"HitRate@{k}"]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#7f8c8d", "#3498db", "#e74c3c"]
    for i, m in enumerate(models):
        values = [results[m][met] for met in metrics]
        ax.bar(x + i * width, values, width, label=m, color=colors[i % 3])
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.002, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Значение метрики")
    ax.set_title(f"Сравнение моделей рекомендательной системы (top-{k})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "models_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"График сравнения сохранён: {out}")


def plot_learning_curve():
    """Кривая обучения LightFM из истории."""
    path = LOGS_DIR / "lightfm_history.json"
    if not path.exists():
        log.warning("Нет lightfm_history.json — кривую обучения не строим")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    hist = data["history"]
    epochs = hist["epoch"]
    train_p = hist["train_precision_at_k"]
    test_p = hist["test_precision_at_k"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_p, marker="o", label="Train P@10", color="#3498db")
    ax.plot(epochs, test_p, marker="s", label="Test P@10", color="#e74c3c")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Precision@10")
    ax.set_title("Кривая обучения LightFM (WARP loss)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "lightfm_learning_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Кривая обучения сохранена: {out}")


# === Main ===

def main():
    log.info("Загрузка артефактов")
    train = load_npz(PROCESSED_DIR / "train_matrix.npz")
    test = load_npz(PROCESSED_DIR / "test_matrix.npz")
    item_features = load_npz(PROCESSED_DIR / "item_features.npz")

    pop_model = joblib.load(MODELS_DIR / "baseline_popularity.joblib")
    utr_model = joblib.load(MODELS_DIR / "baseline_user_top.joblib")
    lightfm_model = joblib.load(MODELS_DIR / "lightfm_model.joblib")
    lightfm_wrapped = LightFMRecommender(lightfm_model, item_features)

    # Сэмпл юзеров с непустым ground truth
    log.info(f"Формирование сэмпла для оценки (target = {EVAL_USERS_SAMPLE})")
    test_csr = test.tocsr()
    users_with_test = np.where(np.diff(test_csr.indptr) > 0)[0]
    log.info(f"Юзеров с непустым TEST: {len(users_with_test):,}")
    rng = np.random.default_rng(RANDOM_SEED)
    if len(users_with_test) > EVAL_USERS_SAMPLE:
        eval_users = rng.choice(users_with_test,
                                size=EVAL_USERS_SAMPLE, replace=False)
    else:
        eval_users = users_with_test
    log.info(f"Сэмпл для оценки: {len(eval_users):,} юзеров")

    results = {}
    results["Popularity"] = evaluate_model(
        "Popularity", pop_model, test_csr, train.tocsr(), eval_users, TOP_K)
    results["UserTopReorder"] = evaluate_model(
        "UserTopReorder", utr_model, test_csr, train.tocsr(), eval_users, TOP_K)
    results["LightFM"] = evaluate_model(
        "LightFM (hybrid)", lightfm_wrapped, test_csr, train.tocsr(), eval_users, TOP_K)
    hybrid_model = HybridRecommender(
        user_top_model=utr_model,
        lightfm_wrapper=lightfm_wrapped,
        train_matrix=train,
        reorder_quota=0.7,
    )
    results["Hybrid"] = evaluate_model(
        "Hybrid (UserTopReorder + LightFM)", hybrid_model,
        test_csr, train.tocsr(), eval_users, TOP_K)

    # Сохранение
    with open(LOGS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "config": {"top_k": TOP_K, "n_eval_users": int(len(eval_users))}
        }, f, ensure_ascii=False, indent=2)

    # Таблица
    df = pd.DataFrame(results).T[[
        "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"
    ]]
    df.columns = [f"Precision@{TOP_K}", f"Recall@{TOP_K}",
                  f"NDCG@{TOP_K}", f"HitRate@{TOP_K}"]
    log.info(f"\nИТОГОВАЯ ТАБЛИЦА:\n{df.round(4).to_string()}")
    df.to_csv(LOGS_DIR / "evaluation_results.csv")

    # Графики
    plot_metrics_comparison(results, TOP_K)
    plot_learning_curve()

    log.info("Оценка завершена")


if __name__ == "__main__":
    main()