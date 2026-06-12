"""
Построение матрицы признаков товаров (item features) для LightFM.
Используем категориальные метаданные: aisle и department.
One-hot кодирование → разреженная матрица.

Размер итоговой матрицы: (n_items, n_items + n_aisles + n_departments).
Первый блок — единичная диагональ (item identity), чтобы модель сохранила
способность учить индивидуальный эмбеддинг товара.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, eye, save_npz
import joblib

from src.config import PROCESSED_DIR
from src.logger import get_logger

log = get_logger("build_features", "preprocess.log")


def main():
    log.info("Загрузка справочника товаров и индексов")
    products = pd.read_parquet(PROCESSED_DIR / "products_enriched.parquet")
    item_to_idx = joblib.load(PROCESSED_DIR / "item_to_idx.joblib")

    # Выравниваем порядок строк по индексу item_to_idx
    products = products.set_index("product_id").loc[item_to_idx.index].reset_index()
    n_items = len(products)
    log.info(f"n_items={n_items}")

    # === One-hot для aisle ===
    aisle_dummies = pd.get_dummies(products["aisle_id"], prefix="aisle", dtype=np.float32)
    aisle_mat = csr_matrix(aisle_dummies.values)
    log.info(f"aisle features: {aisle_mat.shape}")

    # === One-hot для department ===
    dept_dummies = pd.get_dummies(products["department_id"], prefix="dept", dtype=np.float32)
    dept_mat = csr_matrix(dept_dummies.values)
    log.info(f"department features: {dept_mat.shape}")

    # === Identity-блок (каждый товар = свой уникальный признак) ===
    # LightFM требует этого, чтобы модель могла учить индивидуальный эмбеддинг
    identity = eye(n_items, dtype=np.float32, format="csr")

    # Финальная матрица признаков
    item_features = hstack([identity, aisle_mat, dept_mat], format="csr")
    log.info(f"item_features: {item_features.shape} "
             f"(identity={identity.shape[1]} + aisle={aisle_mat.shape[1]} "
             f"+ dept={dept_mat.shape[1]})")

    save_npz(PROCESSED_DIR / "item_features.npz", item_features)
    log.info("✅ Item features сохранены")


if __name__ == "__main__":
    main()