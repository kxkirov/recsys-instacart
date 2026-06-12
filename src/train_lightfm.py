
import time
import json
import numpy as np
from scipy.sparse import load_npz
import joblib
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

from src.config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR,
    LIGHTFM_PARAMS, LIGHTFM_EPOCHS, LIGHTFM_THREADS, TOP_K,
)
from src.logger import get_logger

log = get_logger("train_lightfm", "training.log")


def main():
    log.info("Загрузка артефактов")
    train = load_npz(PROCESSED_DIR / "train_matrix.npz")
    test = load_npz(PROCESSED_DIR / "test_matrix.npz")
    item_features = load_npz(PROCESSED_DIR / "item_features.npz")

    log.info(f"train={train.shape}, test={test.shape}, "
             f"item_features={item_features.shape}")
    log.info(f"Параметры модели: {LIGHTFM_PARAMS}")
    log.info(f"Эпох: {LIGHTFM_EPOCHS}, потоков: {LIGHTFM_THREADS}")

    model = LightFM(**LIGHTFM_PARAMS)

    # Покомпонентное обучение по эпохам — чтобы построить кривую обучения
    history = {"epoch": [], "train_precision_at_k": [],
               "test_precision_at_k": [], "epoch_time_sec": []}

    log.info("=== Старт обучения ===")
    t_total = time.time()
    for epoch in range(1, LIGHTFM_EPOCHS + 1):
        t0 = time.time()
        model.fit_partial(
            train,
            item_features=item_features,
            epochs=1,
            num_threads=LIGHTFM_THREADS,
        )
        epoch_time = time.time() - t0

        # Замеряем метрики не каждую эпоху, а через одну (экономит время)
        if epoch == 1 or epoch % 2 == 0 or epoch == LIGHTFM_EPOCHS:
            train_p = precision_at_k(
                model, train, item_features=item_features,
                k=TOP_K, num_threads=LIGHTFM_THREADS,
            ).mean()
            test_p = precision_at_k(
                model, test, item_features=item_features,
                k=TOP_K, num_threads=LIGHTFM_THREADS,
            ).mean()
        else:
            train_p = float("nan")
            test_p = float("nan")

        history["epoch"].append(epoch)
        history["train_precision_at_k"].append(float(train_p))
        history["test_precision_at_k"].append(float(test_p))
        history["epoch_time_sec"].append(epoch_time)

        log.info(f"Epoch {epoch:02d} | time {epoch_time:5.1f}s | "
                 f"train P@{TOP_K}={train_p:.4f} | test P@{TOP_K}={test_p:.4f}")

    total_time = time.time() - t_total
    log.info(f"=== Обучение завершено за {total_time/60:.1f} мин ===")

    # Финальная оценка модели на TEST (full-basket evaluation, включая reorder)
    log.info("Финальная оценка модели на TEST")
    final_precision = precision_at_k(
        model, test, item_features=item_features,
        k=TOP_K, num_threads=LIGHTFM_THREADS,
    ).mean()
    final_recall = recall_at_k(
        model, test, item_features=item_features,
        k=TOP_K, num_threads=LIGHTFM_THREADS,
    ).mean()
    final_auc = auc_score(
        model, test, item_features=item_features,
        num_threads=LIGHTFM_THREADS,
    ).mean()

    # Сохранение
    joblib.dump(model, MODELS_DIR / "lightfm_model.joblib")
    with open(LOGS_DIR / "lightfm_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "final": {
                "precision_at_k": float(final_precision),
                "recall_at_k": float(final_recall),
                "auc": float(final_auc),
                "total_minutes": total_time / 60,
            },
            "params": LIGHTFM_PARAMS,
            "epochs": LIGHTFM_EPOCHS,
            "top_k": TOP_K,
        }, f, ensure_ascii=False, indent=2)
    log.info("✅ Модель и история обучения сохранены")


if __name__ == "__main__":
    main()