---
title: Recsys Instacart
emoji: 🛒
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# Рекомендательная система Instacart

Прототип гибридной рекомендательной системы для продуктового ритейла на
основе открытого датасета Instacart Market Basket Analysis. Разработано
в рамках выпускной квалификационной работы.

## Архитектура

- **LightFM** — гибридная факторизационная модель с item features (aisle, department)
- **UserTopReorder** — эвристика повторных покупок
- **FP-Growth** — поиск ассоциативных правил
- **Hybrid** — switching-ансамбль (70% reorder + 30% discovery)

## Результаты (TEST, K = 10)

| Модель         | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
| -------------- | ------------ | --------- | ------- | ---------- |
| Popularity     | 0.072        | 0.072     | 0.099   | 0.460      |
| LightFM        | 0.080        | 0.088     | 0.109   | 0.490      |
| UserTopReorder | 0.270        | 0.329     | 0.396   | 0.851      |
| Hybrid         | 0.225        | 0.286     | 0.363   | 0.832      |

## Технологический стек

Python 3.11, LightFM, Streamlit, scipy.sparse, mlxtend, pandas.
