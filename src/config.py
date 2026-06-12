"""
Центральный конфиг проекта.
Все пути, константы и параметры — здесь.
"""
from pathlib import Path

# === Пути ===
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"

# Создаём папки, если их нет
for p in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# === Параметры данных ===
# Случайное зерно для воспроизводимости
RANDOM_SEED = 42

# Сэмплирование пользователей (для ускорения экспериментов на M1)
# None = использовать всех пользователей; число = взять случайный сэмпл
USER_SAMPLE_SIZE = 50_000

# === Параметры модели LightFM ===
LIGHTFM_PARAMS = {
    "no_components": 64,        # размерность латентных векторов
    "loss": "warp",             # лучше всего для top-K ранжирования
    "learning_rate": 0.05,
    "item_alpha": 1e-6,
    "user_alpha": 1e-6,
    "random_state": RANDOM_SEED,
}
LIGHTFM_EPOCHS = 15
LIGHTFM_THREADS = 1  # на M1 OpenMP не работает, оставляем 1

# === Параметры оценки ===
TOP_K = 10  # количество рекомендаций для оценки и UI

# === Параметры ассоциативных правил ===
FPGROWTH_MIN_SUPPORT = 0.003       # товар встречается в >= 0.3% транзакций
FPGROWTH_MIN_CONFIDENCE = 0.1      # P(Y|X) >= 10%
FPGROWTH_SAMPLE_ORDERS = 200_000  # сэмпл корзин для поиска правил