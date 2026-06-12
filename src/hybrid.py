"""
Гибридная рекомендательная стратегия:
  - часть слотов закрывается UserTopReorder (предсказание повторных покупок),
  - часть — LightFM (поиск новых релевантных товаров, discovery).

Архитектурно соответствует таксономии Burke (2002) — switching-стратегия
с разделением ролей: один компонент отвечает за reorder, другой — за discovery.

Обоснование пропорции:
  Reorder Rate ≈ 59% (см. EDA, глава 2 ВКР), поэтому ~70% слотов отдаём
  под reorder-блок (UserTopReorder), остальные 30% — под discovery (LightFM)
  с исключением виденных юзером товаров.
"""
import numpy as np
from scipy.sparse import csr_matrix

from src.config import TOP_K


class HybridRecommender:
    """
    Ансамбль UserTopReorder + LightFM.

    Параметры:
        reorder_quota: доля топ-K, отводимая под повторные покупки.
                       Остаток (k - quota) уходит под discovery.
    """

    def __init__(self, user_top_model, lightfm_wrapper,
                 train_matrix: csr_matrix, reorder_quota: float = 0.7):
        self.user_top = user_top_model
        self.lightfm = lightfm_wrapper
        self.train = train_matrix.tocsr()
        self.reorder_quota = reorder_quota

    def recommend(self, user_idx: int, k: int = TOP_K, exclude_seen=None):
        n_reorder = int(round(k * self.reorder_quota))
        n_discovery = k - n_reorder

        # 1) Reorder-блок: топ повторных покупок юзера
        reorder_recs = self.user_top.recommend(user_idx, k=n_reorder)

        # 2) Discovery-блок: LightFM, но исключаем виденное И уже выбранное в reorder
        seen = set(self.train[user_idx].indices.tolist()) | set(reorder_recs.tolist())
        # Берём с запасом и фильтруем
        lightfm_top = self.lightfm.recommend(user_idx, k=k + len(seen))
        discovery_recs = np.array(
            [i for i in lightfm_top if i not in seen][:n_discovery]
        )

        # Если discovery не набрался — добиваем reorder-ом из юзерской истории
        if len(discovery_recs) < n_discovery:
            extra = self.user_top.recommend(user_idx, k=k * 2)
            extra = [i for i in extra if i not in set(reorder_recs.tolist())
                     and i not in set(discovery_recs.tolist())]
            need = n_discovery - len(discovery_recs)
            discovery_recs = np.concatenate(
                [discovery_recs, np.array(extra[:need], dtype=int)]
            )

        return np.concatenate([reorder_recs, discovery_recs])[:k]