"""
Поиск ассоциативных правил методом FP-Growth (библиотека mlxtend).

Логика:
1. Загружаем заказы из исторических данных (op_prior).
2. Оставляем только заказы наших сэмпл-юзеров (из user_to_idx).
3. Сэмплируем N заказов для FP-Growth (полные данные неподъёмны по памяти).
4. Фильтруем редкие товары (предотвращает комбинаторный взрыв).
5. Sparse one-hot encoding транзакций.
6. FP-Growth → частые наборы товаров.
7. Генерация правил X → Y с метриками support, confidence, lift.
8. Для UI: словарь {product_id: [сопутствующие товары]}, отсортированный по lift.
"""
import pandas as pd
import numpy as np
import joblib
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.config import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, RANDOM_SEED,
    FPGROWTH_MIN_SUPPORT, FPGROWTH_MIN_CONFIDENCE, FPGROWTH_SAMPLE_ORDERS,
)
from src.logger import get_logger

log = get_logger("fpgrowth", "training.log")

# Минимальная частота товара для попадания в анализ
MIN_PRODUCT_FREQ = 150


def main():
    log.info("Загрузка orders + op_prior")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    op_prior = pd.read_csv(RAW_DIR / "order_products__prior.csv")

    orders["order_id"] = orders["order_id"].astype("int32")
    orders["user_id"] = orders["user_id"].astype("int32")
    op_prior["order_id"] = op_prior["order_id"].astype("int32")
    op_prior["product_id"] = op_prior["product_id"].astype("int32")

    log.info("Фильтрация заказов по сэмпл-юзерам")
    user_to_idx = joblib.load(PROCESSED_DIR / "user_to_idx.joblib")
    sampled_users = set(user_to_idx.index.tolist())
    prior_orders = orders[
        (orders["eval_set"] == "prior") & (orders["user_id"].isin(sampled_users))
    ]
    log.info(f"Заказов после фильтрации: {prior_orders.shape[0]:,}")

    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(FPGROWTH_SAMPLE_ORDERS, len(prior_orders))
    sampled_ids = rng.choice(
        prior_orders["order_id"].values, size=sample_size, replace=False
    )
    log.info(f"Сэмпл заказов для FP-Growth: {len(sampled_ids):,}")

    op_sample = op_prior[op_prior["order_id"].isin(set(sampled_ids))]
    log.info(f"Записей в сэмпле: {op_sample.shape[0]:,}")

    # Фильтр редких товаров
    product_counts = op_sample["product_id"].value_counts()
    popular = product_counts[product_counts >= MIN_PRODUCT_FREQ].index
    op_sample = op_sample[op_sample["product_id"].isin(popular)]
    log.info(f"Товаров после фильтра >= {MIN_PRODUCT_FREQ} покупок: {len(popular):,}")

    log.info("Сборка транзакций (заказ = список товаров)")
    transactions = op_sample.groupby("order_id")["product_id"].apply(list).tolist()
    transactions = [t for t in transactions if len(t) >= 2]
    log.info(f"Транзакций с >= 2 товарами: {len(transactions):,}")

    log.info("Sparse one-hot encoding")
    te = TransactionEncoder()
    te_sparse = te.fit(transactions).transform(transactions, sparse=True)
    # mlxtend требует строковые имена колонок при sparse-формате
    str_columns = [str(c) for c in te.columns_]
    df_te = pd.DataFrame.sparse.from_spmatrix(te_sparse, columns=str_columns)
    log.info(f"Матрица транзакций: {df_te.shape}, тип sparse")

    log.info(f"Запуск FP-Growth (min_support={FPGROWTH_MIN_SUPPORT})")
    frequent_itemsets = fpgrowth(
        df_te, min_support=FPGROWTH_MIN_SUPPORT, use_colnames=True
    )
    log.info(f"Найдено частых наборов: {len(frequent_itemsets):,}")

    log.info(f"Генерация правил (min_confidence={FPGROWTH_MIN_CONFIDENCE})")
    rules = association_rules(
        frequent_itemsets, metric="confidence",
        min_threshold=FPGROWTH_MIN_CONFIDENCE,
    )
    rules = rules[rules["lift"] > 1.0]
    log.info(f"Правил с lift>1: {len(rules):,}")

    # Оставляем простые правила (1 товар → 1 товар) для блока "с этим покупают"
    rules["ant_len"] = rules["antecedents"].apply(len)
    rules["con_len"] = rules["consequents"].apply(len)
    simple = rules[(rules["ant_len"] == 1) & (rules["con_len"] == 1)].copy()
    simple["antecedent"] = simple["antecedents"].apply(lambda x: int(list(x)[0]))
    simple["consequent"] = simple["consequents"].apply(lambda x: int(list(x)[0]))
    log.info(f"Простых правил 1→1: {len(simple):,}")

    log.info("Сборка словаря {product_id: [top-K сопутствующих по lift]}")
    simple = simple.sort_values(["antecedent", "lift"], ascending=[True, False])
    item_to_recs = {}
    for ant, group in simple.groupby("antecedent"):
        top = group.nlargest(10, "lift")[
            ["consequent", "support", "confidence", "lift"]
        ].to_dict("records")
        item_to_recs[int(ant)] = top
    log.info(f"Товаров с правилами: {len(item_to_recs):,}")

    joblib.dump(item_to_recs, MODELS_DIR / "fpgrowth_rules.joblib")
    simple[["antecedent", "consequent", "support", "confidence", "lift"]].to_parquet(
        PROCESSED_DIR / "fpgrowth_rules.parquet"
    )
    log.info("✅ FP-Growth правила сохранены")

    # Демо: первые 3 товара с правилами
    products = pd.read_parquet(PROCESSED_DIR / "products_enriched.parquet")
    pmap = products.set_index("product_id")["product_name"].to_dict()
    for k in list(item_to_recs.keys())[:3]:
        recs = item_to_recs[k][:3]
        log.info(f"  {pmap.get(k, k)} → " +
                 ", ".join(f"{pmap.get(r['consequent'], r['consequent'])} "
                           f"(lift={r['lift']:.2f})" for r in recs))


if __name__ == "__main__":
    main()