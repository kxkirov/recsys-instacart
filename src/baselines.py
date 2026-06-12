"""
Базовые рекомендательные модели для сравнения с LightFM.

1. PopularityRecommender — топ-K самых популярных товаров (одинаков для всех).
2. UserTopReorderRecommender — топ-K товаров из истории конкретного пользователя,
   отсортированных по частоте покупок.

Обе модели реализуют единый интерфейс recommend(user_idx, k) → list[item_idx],
что упрощает дальнейшее сравнение через единый evaluator.
"""
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import joblib

from src.config import PROCESSED_DIR, MODELS_DIR, TOP_K
from src.logger import get_logger

log = get_logger("baselines", "training.log")


class PopularityRecommender:
    """Глобальный топ — самые популярные товары по числу покупок в TRAIN."""

    def __init__(self):
        self.popular_items_ = None  # отсортированные индексы товаров по убыванию частоты

    def fit(self, train_matrix: csr_matrix):
        item_counts = np.asarray(train_matrix.sum(axis=0)).ravel()
        self.popular_items_ = np.argsort(-item_counts)
        log.info(f"PopularityRecommender обучен: топ-3 индекса = "
                 f"{self.popular_items_[:3].tolist()}")
        return self

    def recommend(self, user_idx: int, k: int = TOP_K,
                  exclude_seen: csr_matrix = None) -> np.ndarray:
        if exclude_seen is None:
            return self.popular_items_[:k]
        seen = set(exclude_seen[user_idx].indices.tolist())
        result = [i for i in self.popular_items_ if i not in seen][:k]
        return np.array(result)


class UserTopReorderRecommender:
    """
    Персональная популярность. Для каждого юзера сортируем его историю
    по частоте покупок (в TRAIN) и берём top-K. Это сильный бейзлайн
    для продуктового ритейла, поскольку Reorder Rate ≈ 59%.
    """

    def __init__(self):
        self.train_matrix_ = None
        self.global_popular_ = None  # fallback для новых юзеров

    def fit(self, train_matrix: csr_matrix):
        self.train_matrix_ = train_matrix.tocsr()
        item_counts = np.asarray(train_matrix.sum(axis=0)).ravel()
        self.global_popular_ = np.argsort(-item_counts)
        log.info("UserTopReorderRecommender обучен")
        return self

    def recommend(self, user_idx: int, k: int = TOP_K,
                  exclude_seen: csr_matrix = None) -> np.ndarray:
        row = self.train_matrix_[user_idx]
        if row.nnz == 0:
            # холодный пользователь → fallback на глобальный топ
            return self.global_popular_[:k]
        items = row.indices
        counts = row.data
        order = np.argsort(-counts)
        result = items[order][:k]
        # если истории < k, добиваем глобальным топом
        if len(result) < k:
            seen = set(result.tolist())
            extra = [i for i in self.global_popular_ if i not in seen]
            result = np.concatenate([result, extra[: k - len(result)]])
        return result


def main():
    log.info("Загрузка train-матрицы")
    train = load_npz(PROCESSED_DIR / "train_matrix.npz")

    log.info("Обучение PopularityRecommender")
    pop = PopularityRecommender().fit(train)
    joblib.dump(pop, MODELS_DIR / "baseline_popularity.joblib")

    log.info("Обучение UserTopReorderRecommender")
    utr = UserTopReorderRecommender().fit(train)
    joblib.dump(utr, MODELS_DIR / "baseline_user_top.joblib")

    # Sanity-check: смотрим рекомендации для юзера 0
    log.info(f"Демо Popularity для user_idx=0: {pop.recommend(0, k=5).tolist()}")
    log.info(f"Демо UserTopReorder для user_idx=0: {utr.recommend(0, k=5).tolist()}")
    log.info("✅ Baseline-модели обучены и сохранены")


if __name__ == "__main__":
    main()