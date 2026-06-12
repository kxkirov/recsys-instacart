"""
Загрузка сырых CSV Instacart, очистка, оптимизация типов,
формирование train/test разбиения и разреженной user-item матрицы.

Логика разбиения:
- Для каждого пользователя его ПОСЛЕДНИЙ заказ (eval_set='train') идёт в TEST.
- Вся остальная история (eval_set='prior') идёт в TRAIN.
- Это эмулирует реальный сценарий: "по истории предсказать следующую корзину".
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import joblib

from src.config import (
    RAW_DIR, PROCESSED_DIR, RANDOM_SEED, USER_SAMPLE_SIZE
)
from src.logger import get_logger

log = get_logger("preprocess", "preprocess.log")


def load_raw():
    log.info("Загрузка сырых CSV из data/raw/")
    aisles = pd.read_csv(RAW_DIR / "aisles.csv")
    departments = pd.read_csv(RAW_DIR / "departments.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    op_prior = pd.read_csv(RAW_DIR / "order_products__prior.csv")
    op_train = pd.read_csv(RAW_DIR / "order_products__train.csv")
    log.info(f"orders={orders.shape}, op_prior={op_prior.shape}, "
             f"op_train={op_train.shape}, products={products.shape}")
    return aisles, departments, products, orders, op_prior, op_train


def optimize_types(orders, products, op_prior, op_train):
    """Downcasting для экономии памяти (логика из EDA, глава 2)."""
    log.info("Оптимизация типов данных")

    orders["eval_set"] = orders["eval_set"].astype("category")
    orders["order_id"] = orders["order_id"].astype("int32")
    orders["user_id"] = orders["user_id"].astype("int32")
    orders["order_number"] = orders["order_number"].astype("int16")
    orders["order_dow"] = orders["order_dow"].astype("int8")
    orders["order_hour_of_day"] = orders["order_hour_of_day"].astype("int8")
    orders["days_since_prior_order"] = (
        orders["days_since_prior_order"].fillna(0).astype("float32")
    )

    products["product_id"] = products["product_id"].astype("int32")
    products["aisle_id"] = products["aisle_id"].astype("int16")
    products["department_id"] = products["department_id"].astype("int16")

    for df in (op_prior, op_train):
        df["order_id"] = df["order_id"].astype("int32")
        df["product_id"] = df["product_id"].astype("int32")
        df["add_to_cart_order"] = df["add_to_cart_order"].astype("int16")
        df["reordered"] = df["reordered"].astype("int8")

    return orders, products, op_prior, op_train


def sample_users(orders, n_users):
    """Случайный сэмпл пользователей для ускорения на M1."""
    if n_users is None:
        log.info("Используем всех пользователей (без сэмплинга)")
        return orders
    rng = np.random.default_rng(RANDOM_SEED)
    all_users = orders["user_id"].unique()
    sampled = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)
    sampled_set = set(sampled.tolist())
    filtered = orders[orders["user_id"].isin(sampled_set)].copy()
    log.info(f"Сэмпл: {len(sampled_set)} пользователей из {len(all_users)}, "
             f"orders {orders.shape[0]} → {filtered.shape[0]}")
    return filtered


def build_train_test(orders, op_prior, op_train):
    """
    TRAIN: все взаимодействия из истории (eval_set='prior').
    TEST: последний заказ пользователя (eval_set='train' в датасете Instacart).
    Пользователи с eval_set='test' (без размеченного последнего заказа)
    из тестовой выборки исключаются.
    """
    log.info("Формирование train/test")

    prior_orders = orders[orders["eval_set"] == "prior"][["order_id", "user_id"]]
    train_orders = orders[orders["eval_set"] == "train"][["order_id", "user_id"]]

    # TRAIN: история покупок
    train_df = op_prior.merge(prior_orders, on="order_id", how="inner")
    train_df = train_df[["user_id", "product_id", "reordered", "add_to_cart_order"]]

    # TEST: последний заказ
    test_df = op_train.merge(train_orders, on="order_id", how="inner")
    test_df = test_df[["user_id", "product_id", "reordered", "add_to_cart_order"]]

    log.info(f"TRAIN interactions: {train_df.shape[0]:,}, "
             f"unique users: {train_df['user_id'].nunique():,}")
    log.info(f"TEST interactions: {test_df.shape[0]:,}, "
             f"unique users: {test_df['user_id'].nunique():,}")
    return train_df, test_df


def build_id_mappings(train_df, products):
    """Сквозные индексы 0..N для матрицы (user_id и product_id могут иметь пропуски)."""
    log.info("Построение индексов user/item")
    user_ids = np.sort(train_df["user_id"].unique())
    item_ids = np.sort(products["product_id"].unique())

    user_to_idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    item_to_idx = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    log.info(f"n_users={len(user_ids):,}, n_items={len(item_ids):,}")
    return user_to_idx, item_to_idx


def build_interaction_matrix(df, user_to_idx, item_to_idx, name="train"):
    """
    Разреженная матрица user × item.
    Значение = количество покупок товара пользователем (неявная обратная связь).
    """
    log.info(f"Сборка матрицы взаимодействий [{name}]")
    df = df[df["user_id"].isin(user_to_idx.index)
            & df["product_id"].isin(item_to_idx.index)].copy()

    rows = user_to_idx.loc[df["user_id"].values].values
    cols = item_to_idx.loc[df["product_id"].values].values
    data = np.ones(len(df), dtype=np.float32)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    # суммирование дубликатов (если юзер брал товар несколько раз)
    mat.sum_duplicates()

    nnz = mat.nnz
    sparsity = 1 - nnz / (n_users * n_items)
    log.info(f"[{name}] shape={mat.shape}, nnz={nnz:,}, sparsity={sparsity:.5f}")
    return mat


def main():
    aisles, departments, products, orders, op_prior, op_train = load_raw()
    orders, products, op_prior, op_train = optimize_types(
        orders, products, op_prior, op_train
    )

    orders = sample_users(orders, USER_SAMPLE_SIZE)

    train_df, test_df = build_train_test(orders, op_prior, op_train)
    user_to_idx, item_to_idx = build_id_mappings(train_df, products)

    train_mat = build_interaction_matrix(train_df, user_to_idx, item_to_idx, "train")
    test_mat = build_interaction_matrix(test_df, user_to_idx, item_to_idx, "test")

    # Справочник товаров для UI и интерпретации
    products_enriched = (
        products.merge(aisles, on="aisle_id", how="left")
                .merge(departments, on="department_id", how="left")
    )

    log.info("Сохранение артефактов в data/processed/")
    save_npz(PROCESSED_DIR / "train_matrix.npz", train_mat)
    save_npz(PROCESSED_DIR / "test_matrix.npz", test_mat)
    joblib.dump(user_to_idx, PROCESSED_DIR / "user_to_idx.joblib")
    joblib.dump(item_to_idx, PROCESSED_DIR / "item_to_idx.joblib")
    products_enriched.to_parquet(PROCESSED_DIR / "products_enriched.parquet")
    train_df.to_parquet(PROCESSED_DIR / "train_interactions.parquet")
    test_df.to_parquet(PROCESSED_DIR / "test_interactions.parquet")

    log.info("✅ Preprocess завершён")


if __name__ == "__main__":
    main()