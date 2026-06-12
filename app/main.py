"""Streamlit-интерфейс рекомендательной системы Instacart."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
import joblib

from src.config import PROCESSED_DIR, MODELS_DIR, TOP_K
from src.baselines import PopularityRecommender, UserTopReorderRecommender  # noqa
from src.hybrid import HybridRecommender


# ============================================================
# Загрузка артефактов
# ============================================================

@st.cache_resource(show_spinner="Загрузка моделей...")
def load_artifacts():
    train = load_npz(PROCESSED_DIR / "train_matrix.npz")
    item_features = load_npz(PROCESSED_DIR / "item_features.npz")
    user_to_idx = joblib.load(PROCESSED_DIR / "user_to_idx.joblib")
    item_to_idx = joblib.load(PROCESSED_DIR / "item_to_idx.joblib")
    products = pd.read_parquet(PROCESSED_DIR / "products_enriched.parquet")
    fp_rules = joblib.load(MODELS_DIR / "fpgrowth_rules.joblib")
    pop_model = joblib.load(MODELS_DIR / "baseline_popularity.joblib")
    utr_model = joblib.load(MODELS_DIR / "baseline_user_top.joblib")
    lightfm_model = joblib.load(MODELS_DIR / "lightfm_model.joblib")
    idx_to_item = pd.Series(item_to_idx.index.values, index=item_to_idx.values)
    return dict(
        train=train, item_features=item_features,
        user_to_idx=user_to_idx, item_to_idx=item_to_idx, idx_to_item=idx_to_item,
        products=products.set_index("product_id"),
        fp_rules=fp_rules,
        pop_model=pop_model, utr_model=utr_model, lightfm_model=lightfm_model,
    )


class LightFMRecommender:
    def __init__(self, model, item_features):
        self.model = model
        self.item_features = item_features
        self._all_items = np.arange(item_features.shape[0])

    def recommend(self, user_idx, k=TOP_K, exclude_seen=None):
        scores = self.model.predict(
            user_ids=int(user_idx), item_ids=self._all_items,
            item_features=self.item_features,
        )
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])]


# ============================================================
# Утилиты
# ============================================================

def get_user_history(art, user_idx, top_n):
    row = art["train"][user_idx]
    if row.nnz == 0:
        return pd.DataFrame(columns=["product_id", "product_name",
                                     "department", "purchases"])
    items_idx = row.indices
    counts = row.data.astype(int)
    order = np.argsort(-counts)[:top_n]
    pids = art["idx_to_item"].loc[items_idx[order]].values
    info = art["products"].loc[pids][["product_name", "department"]]
    info = info.reset_index().rename(columns={"index": "product_id"})
    info["purchases"] = counts[order]
    return info


def get_recommendations(model_name, art, user_idx, k, allowed_departments=None):
    n_ask = k * 5 if allowed_departments else k

    if model_name == "Popularity":
        recs = art["pop_model"].recommend(user_idx, k=n_ask)
    elif model_name == "UserTopReorder":
        recs = art["utr_model"].recommend(user_idx, k=n_ask)
    elif model_name == "LightFM":
        recs = LightFMRecommender(
            art["lightfm_model"], art["item_features"]
        ).recommend(user_idx, k=n_ask)
    elif model_name == "Hybrid":
        lfm = LightFMRecommender(art["lightfm_model"], art["item_features"])
        hybrid = HybridRecommender(art["utr_model"], lfm, art["train"], 0.7)
        recs = hybrid.recommend(user_idx, k=n_ask)
    else:
        raise ValueError(model_name)

    pids = art["idx_to_item"].loc[recs].values
    df = art["products"].loc[pids][["product_name", "department"]]
    df = df.reset_index().rename(columns={"index": "product_id"})
    if allowed_departments:
        df = df[df["department"].isin(allowed_departments)]
    return df.head(k).reset_index(drop=True)


def get_fbt(art, history_pids, max_anchors=3, top_per_anchor=3):
    out = []
    rules = art["fp_rules"]
    for pid in history_pids[:max_anchors]:
        if pid not in rules:
            continue
        anchor_name = art["products"].loc[pid, "product_name"]
        for r in rules[pid][:top_per_anchor]:
            cons_pid = r["consequent"]
            if cons_pid in art["products"].index:
                out.append({
                    "Якорный товар": anchor_name,
                    "Рекомендуем": art["products"].loc[cons_pid, "product_name"],
                    "Отдел": art["products"].loc[cons_pid, "department"],
                    "Lift": round(r["lift"], 2),
                    "Confidence": round(r["confidence"], 3),
                })
    return pd.DataFrame(out)


def render_recs_table(df, height=420):
    if df.empty:
        st.info("Нет данных под текущие фильтры.")
        return
    view = df[["product_name", "department"]].rename(columns={
        "product_name": "Товар", "department": "Отдел",
    })
    view.index = view.index + 1
    st.dataframe(view, use_container_width=True, height=height)


def render_history_table(df, height=420):
    if df.empty:
        st.info("История пуста.")
        return
    view = df[["product_name", "department", "purchases"]].rename(columns={
        "product_name": "Товар", "department": "Отдел", "purchases": "Куплено",
    })
    view.index = view.index + 1
    st.dataframe(view, use_container_width=True, height=height)


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Recsys Instacart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Минимальный CSS — типографика и аккуратные карточки
st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    h1 {
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    h2 {
        font-weight: 600;
        font-size: 1.4rem;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        color: #e8e8e8;
    }
    h3 {
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 0.6rem;
        color: #d0d0d0;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        opacity: 0.7;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    hr { margin: 1.8rem 0; opacity: 0.2; }
    .stExpander { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("Рекомендательная система Instacart")

art = load_artifacts()
all_user_ids = sorted(art["user_to_idx"].index.tolist())
all_departments = sorted(art["products"]["department"].dropna().unique().tolist())

# ----- Sidebar -----
with st.sidebar:
    st.subheader("Параметры")

    if "selected_user" not in st.session_state:
        st.session_state.selected_user = int(all_user_ids[0])

    user_input = st.number_input(
        "ID пользователя",
        min_value=int(min(all_user_ids)),
        max_value=int(max(all_user_ids)),
        value=int(st.session_state.selected_user), step=1,
    )
    st.session_state.selected_user = int(user_input)

    if st.button("Случайный пользователь", use_container_width=True):
        st.session_state.selected_user = int(np.random.choice(all_user_ids))
        st.rerun()

    st.divider()

    top_k = st.slider("Размер списка рекомендаций", 5, 15, 10)

    dept_filter = st.multiselect(
        "Фильтр по отделам",
        options=all_departments,
        default=[],
        help="Если пусто — без фильтра",
    )

    st.divider()
    st.caption("Информация о системе")
    st.caption(f"Пользователей в обучении: {len(all_user_ids):,}")
    st.caption(f"Товаров в каталоге: {len(art['products']):,}")
    st.caption(f"FP-Growth правил: {len(art['fp_rules']):,}")


# ----- Main -----
user_id = st.session_state.selected_user
if user_id not in art["user_to_idx"].index:
    st.error(f"Пользователь {user_id} отсутствует в выборке.")
    st.stop()

user_idx = int(art["user_to_idx"].loc[user_id])
history_full = get_user_history(art, user_idx, top_n=50)
total_items = int(art["train"][user_idx].sum())
unique_items = int(art["train"][user_idx].nnz)
fav_dept = (history_full.groupby("department")["purchases"].sum()
            .idxmax() if not history_full.empty else "—")

# Метрики юзера
c1, c2, c3, c4 = st.columns(4)
c1.metric("ID пользователя", f"#{user_id}")
c2.metric("Куплено товаров (всего)", f"{total_items:,}")
c3.metric("Уникальных позиций", f"{unique_items:,}")
c4.metric("Любимый отдел", fav_dept)

st.divider()

# === История покупок (полная ширина) ===
st.header("История покупок")
st.caption(f"Топ-{top_k} самых частых товаров пользователя в обучающей выборке.")
render_history_table(history_full.head(top_k), height=min(420, 50 + top_k * 36))

st.divider()

# === Три модели в три колонки ===
st.header("Рекомендации")
st.caption("Топ-список от каждой модели. Hybrid — основная архитектура: "
           "70% слотов отдаются под повторные покупки (UserTopReorder), "
           "30% — под discovery новых товаров (LightFM).")

col_hyb, col_utr, col_lfm = st.columns(3, gap="medium")

table_height = min(420, 50 + top_k * 36)

with col_hyb:
    st.subheader("Hybrid")
    st.caption("Ансамбль UserTopReorder + LightFM")
    df = get_recommendations("Hybrid", art, user_idx, k=top_k,
                             allowed_departments=dept_filter or None)
    render_recs_table(df, height=table_height)

with col_utr:
    st.subheader("UserTopReorder")
    st.caption("Эвристика по истории пользователя")
    df = get_recommendations("UserTopReorder", art, user_idx, k=top_k,
                             allowed_departments=dept_filter or None)
    render_recs_table(df, height=table_height)

with col_lfm:
    st.subheader("LightFM")
    st.caption("Гибридная факторизационная модель")
    df = get_recommendations("LightFM", art, user_idx, k=top_k,
                             allowed_departments=dept_filter or None)
    render_recs_table(df, height=table_height)

st.divider()

# === FP-Growth ===
st.header("Часто покупают вместе")
st.caption("Ассоциативные правила, найденные методом FP-Growth для топ-3 товаров "
           "из истории пользователя. Lift > 1 означает, что товары появляются "
           "вместе чаще, чем случайно.")

if history_full.empty:
    st.info("История пуста.")
else:
    fbt = get_fbt(art, history_full["product_id"].tolist())
    if fbt.empty:
        st.info("Для топ-товаров истории не найдено правил с lift > 1.")
    else:
        st.dataframe(fbt, hide_index=True, use_container_width=True, height=240)

# === Контрольный baseline (Popularity) ===
with st.expander("Контрольный baseline — Popularity"):
    st.caption("Модель возвращает одни и те же популярные товары всем пользователям "
               "независимо от истории. Используется как нижняя граница качества.")
    pop_df = get_recommendations("Popularity", art, user_idx, k=top_k,
                                 allowed_departments=dept_filter or None)
    render_recs_table(pop_df, height=table_height)