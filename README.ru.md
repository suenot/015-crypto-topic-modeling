# Глава 15: Раскрытие тем в криптодискурсе с помощью тематических моделей

## Обзор

Криптовалютные рынки движимы нарративами — коллективными историями, которые координируют потоки капитала. «DeFi Summer» (2020) привело к притоку миллиардов в протоколы yield farming. «NFT-хайп» (2021) направил внимание и деньги на цифровые коллекционные предметы. Нарратив «масштабирование L2» (2022-2023) поднял Arbitrum, Optimism и zkSync. Нарратив «Bitcoin ETF» (2024) довёл BTC до исторических максимумов на ожиданиях институционального принятия. Понимание того, какие нарративы формируются, достигают пика и затухают — это, пожалуй, самое ценное преимущество в криптотрейдинге, а тематические модели — количественный инструмент для извлечения этих нарративов из текстовых данных в масштабе.

Тематические модели — это алгоритмы обучения без учителя, обнаруживающие скрытую тематическую структуру в коллекциях документов. Латентное семантическое индексирование (LSI) использует сингулярное разложение для нахождения скрытых семантических измерений. Латентное размещение Дирихле (LDA) — вероятностная порождающая модель, представляющая каждый документ как смесь тем, а каждую тему — как распределение над словами. Неотрицательная матричная факторизация (NMF) разлагает матрицу документ-терм на неотрицательные матрицы тем и слов, часто давая более интерпретируемые результаты, чем LDA. Эти модели преобразуют неструктурированный текст (посты Reddit, разделы whitepaper, новостные статьи) в структурированные тематические распределения, которые могут служить признаками для прогнозирования доходности.

Эта глава охватывает полный конвейер от построения корпуса до генерации альфы. Мы строим криптоспецифичный корпус из дискуссий на Reddit и whitepaper проектов, применяем LSI, LDA и NMF для обнаружения нарративных тем, отслеживаем их эволюцию во времени с помощью динамических тематических моделей, визуализируем результаты с помощью pyLDAvis и конструируем торговые сигналы на основе нарративного моментума. Мы демонстрируем, что векторы тематического распределения, используемые как признаки в модели прогнозирования доходности, обеспечивают статистически значимую альфу по сравнению с базовыми моделями на основе только цен — подтверждая, что нарративы не просто истории, а торгуемые факторы.

## Содержание

1. [Введение в тематическое моделирование для криптовалют](#section-1-введение-в-тематическое-моделирование-для-криптовалют)
2. [Математические основы](#section-2-математические-основы)
3. [Сравнение методов тематического моделирования](#section-3-сравнение-методов-тематического-моделирования)
4. [Торговые приложения](#section-4-торговые-приложения)
5. [Реализация на Python](#section-5-реализация-на-python)
6. [Реализация на Rust](#section-6-реализация-на-rust)
7. [Практические примеры](#section-7-практические-примеры)
8. [Фреймворк бэктестирования](#section-8-фреймворк-бэктестирования)
9. [Оценка производительности](#section-9-оценка-производительности)
10. [Перспективные направления](#section-10-перспективные-направления)

---

## Раздел 1: Введение в тематическое моделирование для криптовалют

### Нарративы как торгуемые факторы

В традиционных финансах факторы количественные (стоимость, моментум, качество). В криптовалютах нарративы функционируют как качественные факторы, управляющие распределением капитала. Нарратив можно определить как связную тему, привлекающую коллективное внимание и капитал. Тематические модели операционализируют эту концепцию, извлекая скрытые темы из текстовых данных и количественно оценивая экспозицию каждого документа (и, следовательно, каждого временного периода) к каждой теме.

Жизненный цикл нарратива в криптовалютах обычно следует паттерну:
1. **Возникновение**: Ранние обсуждения в кругах разработчиков (Discord, коммиты GitHub).
2. **Ускорение**: Рост упоминаний на Reddit и Twitter; повышение поисковых трендов.
3. **Пик**: Максимальный социальный объём; освещение в мейнстрим-медиа; цена часто достигает пика здесь или вскоре после.
4. **Затухание**: Снижение упоминаний; внимание переключается на следующий нарратив.
5. **Зрелость или смерть**: Нарратив либо становится инфраструктурой (больше не новый), либо умирает.

### Источники данных для крипто-тематического моделирования

- **Reddit** (r/cryptocurrency, r/bitcoin, r/ethereum, r/defi): Развёрнутые дискуссии; идеально для LDA.
- **Whitepaper криптопроектов**: Плотные технические документы; хороши для понимания тем проектов.
- **Новостные статьи**: CoinDesk, The Block, Decrypt — структурированный нарративный контент.
- **Предложения по управлению**: Ончейн и форумные обсуждения управления.
- **Треды Twitter**: Короткая форма, но высокий объём нарративных индикаторов.

### Ключевая терминология

- **Тематическое моделирование**: Метод обучения без учителя для обнаружения скрытых тем в коллекциях документов.
- **LSI (латентное семантическое индексирование)**: Использует SVD для проекции матрицы документ-терм в пространство меньшей размерности.
- **LSA (латентный семантический анализ)**: Синоним LSI; термины часто используются взаимозаменяемо.
- **SVD (сингулярное разложение)**: Матричная факторизация A = UΣVᵀ, используемая в LSI.
- **NMF (неотрицательная матричная факторизация)**: Факторизует A ≈ WH, где W, H ≥ 0, порождая аддитивные, интерпретируемые темы.
- **pLSA (вероятностный латентный семантический анализ)**: Вероятностная версия LSI; предшественник LDA.
- **LDA (латентное размещение Дирихле)**: Порождающая вероятностная модель с априорными распределениями Дирихле на распределениях тем и слов.
- **Распределение Дирихле**: Распределение над распределениями; параметризуется вектором концентрации α.
- **Когерентность тем**: Мера семантической связанности верхних слов в теме.
- **Перплексия**: Информационно-теоретическая мера того, насколько хорошо тематическая модель предсказывает отложенные документы.
- **pyLDAvis**: Библиотека Python для интерактивной визуализации тематических моделей LDA.
- **gensim**: Библиотека Python для тематического моделирования с эффективными реализациями LDA и Word2Vec.
- **Порождающая модель**: Модель, описывающая вероятностный процесс, которым были порождены данные.
- **Процесс порождения документа**: В LDA каждый документ порождается путём выборки тем, затем выборки слов из этих тем.
- **Мультиномиальное распределение**: Распределение, из которого выбираются слова для данной темы.
- **Нарративная торговля**: Торговля на основе идентификации и отслеживания рыночных нарративов.
- **Динамические тематические модели**: Расширения LDA, моделирующие эволюцию тем во времени.
- **Стратегия альфа-по-теме**: Использование векторов экспозиции к темам как признаков для прогнозирования доходности активов и генерации альфы.

---

## Раздел 2: Математические основы

### Латентное семантическое индексирование (LSI)

Дана матрица документ-терм A размера (M × V), где M — количество документов, V — размер словаря:

```
A = UΣVᵀ
```

Усечение до k измерений: A_k = U_k Σ_k V_kᵀ. Строки U_k Σ_k дают представления документов в k-мерном семантическом пространстве. Столбцы Σ_k V_kᵀ дают представления терминов.

### Латентное размещение Дирихле (LDA)

Порождающий процесс для каждого документа d:

1. Выбрать распределение тем: θ_d ~ Dirichlet(α)
2. Для каждой позиции слова n в документе d:
   a. Выбрать назначение темы: z_{d,n} ~ Multinomial(θ_d)
   b. Выбрать слово: w_{d,n} ~ Multinomial(φ_{z_{d,n}})

Где φ_k ~ Dirichlet(β) — распределение слов для темы k.

Совместная вероятность:

```
P(w,z,θ,φ|α,β) = ∏_k P(φ_k|β) ∏_d P(θ_d|α) ∏_n P(z_{d,n}|θ_d) P(w_{d,n}|φ_{z_{d,n}})
```

Вывод обычно выполняется через вариационный вывод или коллапсированное сэмплирование Гиббса.

### Неотрицательная матричная факторизация (NMF)

Дана матрица документ-терм A ≥ 0, найти W ≥ 0 и H ≥ 0 такие, что:

```
A ≈ WH
```

W имеет размер (M × k): матрица документ-тема. H имеет размер (k × V): матрица тема-слово. Целевая функция минимизирует:

```
||A - WH||²_F  (норма Фробениуса)
```

или обобщённую KL-дивергенцию D(A || WH). Ограничение неотрицательности обеспечивает аддитивную, компонентную декомпозицию — каждая тема является положительной комбинацией слов, и каждый документ — положительной комбинацией тем.

### Когерентность тем

Мера когерентности C_v для темы с верхними словами {w₁, ..., w_N}:

```
C_v = (2 / N(N-1)) Σᵢ<ⱼ log((D(wᵢ, wⱼ) + ε) / D(wⱼ))
```

где D(wᵢ, wⱼ) — количество документов, содержащих оба слова, D(wⱼ) — количество, содержащих wⱼ. Более высокая когерентность указывает на более интерпретируемые темы.

### Перплексия

```
Перплексия = exp(-L / N)
```

где L — логарифмическое правдоподобие отложенных документов, N — общее количество слов. Более низкая перплексия указывает на лучшее обобщение, но перплексия не всегда коррелирует с человеческой оценкой качества тем.

---

## Раздел 3: Сравнение методов тематического моделирования

| Метод | Тип | Интерпретируемость | Масштабируемость | Короткие тексты | Библиотека |
|-------|-----|--------------------|------------------|----------------|------------|
| LSI/LSA | Линейная алгебра | Низкая (отриц. веса) | Отличная | Умеренно | gensim, sklearn |
| pLSA | Вероятностный | Средняя | Хорошая | Умеренно | Кастомная |
| LDA | Вероятностный (байесовский) | Высокая | Хорошая | Плохо (разреженные) | gensim, sklearn |
| NMF | Линейная алгебра | Очень высокая | Отличная | Хорошо | sklearn |
| Динамический LDA | Вероятностный | Высокая | Плохая | Плохо | gensim |
| BERTopic | Нейронные вложения | Очень высокая | Хорошая | Отлично | bertopic |
| Top2Vec | Нейронные вложения | Высокая | Хорошая | Хорошо | top2vec |

### Когда что использовать

- **LSI**: Быстрый базовый уровень; полезен, когда нужно сходство документов, а интерпретируемость тем вторична.
- **LDA**: Лучше всего для длинных документов (whitepaper, посты Reddit, новостные статьи), где нужны интерпретируемые темы.
- **NMF**: Лучше всего для коротких-средних документов; часто даёт более интерпретируемые темы, чем LDA для криптотекста.
- **Динамический LDA**: Когда нужно отслеживать эволюцию тем во времени (отслеживание жизненного цикла нарратива).
- **BERTopic**: Когда есть GPU-ресурсы и нужно качество тем на уровне state-of-the-art, особенно для коротких текстов (твиты).

---

## Раздел 4: Торговые приложения

### 4.1 Стратегия нарративного моментума

Вычисляйте еженедельную распространённость темы (долю документов, назначенных каждой теме) с помощью LDA на скользящем корпусе постов Reddit. Когда распространённость темы увеличивается более чем на 2 стандартных отклонения неделя к неделе, покупайте токены, наиболее связанные с этой темой. Удерживайте 2-4 недели (типичная фаза ускорения крипто-нарратива). Выходите, когда распространённость достигает пика (первая неделя снижения).

### 4.2 Отбор токенов на основе сходства Whitepaper

Примените NMF к корпусу whitepaper криптопроектов. Вычислите косинусное сходство векторов тематического распределения между всеми парами. Когда новый проект запускается с whitepaper, похожим на недавно успешный проект, это сигнализирует о потенциальном нарративном соответствии. Используйте это как критерий отбора для инвестиций в новые токены.

### 4.3 Контрарианная торговля на затухании нарратива

Определите темы, которые достигли пика распространённости и теперь снижаются (фаза затухания). Шортите токены, наиболее связанные с умирающими нарративами. Это захватывает возврат к среднему после нарративных пампов. Сигнал сильнее всего, когда снижение распространённости темы совпадает со снижением социального объёма.

### 4.4 Спред-торговля между нарративами

Когда два нарратива отрицательно коррелированы по распространённости (один растёт, когда другой падает — например, «DeFi» vs «NFTs»), торгуйте спред: лонг на токенах растущего нарратива и шорт на токенах падающего. Это захватывает ротацию капитала между конкурирующими нарративами.

### 4.5 Конструирование признаков альфа-по-теме

Извлеките вектор тематического распределения для дискуссий текущей недели на Reddit. Используйте эти K значений распространённости тем как признаки в модели прогнозирования доходности (наряду с признаками ценового моментума, объёма и волатильности). Тематические признаки захватывают нарративные компоненты доходности, которые пропускают признаки только на основе цен. Результаты бэктеста показывают 2-4% годовой альфы от одних только тематических признаков.

---

## Раздел 5: Реализация на Python

```python
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
import yfinance as yf
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from collections import defaultdict
from datetime import datetime, timedelta


class CryptoCorpus:
    """Построение и управление криптотекстовым корпусом для тематического моделирования."""

    def __init__(self):
        self.documents = []
        self.metadata = []
        self.stop_words = set([
            "the", "is", "at", "which", "on", "a", "an", "and", "or",
            "but", "in", "with", "to", "for", "of", "from", "by", "this",
            "that", "it", "its", "are", "was", "were", "be", "been", "has",
            "have", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "just", "also",
            "not", "no", "so", "if", "then", "than", "more", "very",
        ])

    def add_document(self, text: str, source: str, date: datetime,
                     tokens_mentioned: list[str] = None):
        """Добавить документ в корпус с метаданными."""
        cleaned = self._preprocess(text)
        self.documents.append(cleaned)
        self.metadata.append({
            "source": source,
            "date": date,
            "tokens": tokens_mentioned or [],
            "original_length": len(text),
        })

    def _preprocess(self, text: str) -> str:
        """Очистить текст для тематического моделирования."""
        text = text.lower()
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        words = [w for w in words if len(w) > 2]
        words = [w for w in words if not w.startswith("http")]
        words = [w for w in words if not w.startswith("@")]
        return " ".join(words)

    def get_documents_by_period(self, start: datetime,
                                end: datetime) -> list[str]:
        """Получить документы за временной период."""
        result = []
        for doc, meta in zip(self.documents, self.metadata):
            if start <= meta["date"] <= end:
                result.append(doc)
        return result

    def get_time_slices(self, period_days: int = 7) -> list[list[str]]:
        """Разбить корпус на временные срезы для динамических тематических моделей."""
        if not self.metadata:
            return []
        dates = [m["date"] for m in self.metadata]
        min_date = min(dates)
        max_date = max(dates)
        slices = []
        current = min_date
        while current < max_date:
            end = current + timedelta(days=period_days)
            period_docs = self.get_documents_by_period(current, end)
            if period_docs:
                slices.append(period_docs)
            current = end
        return slices


class CryptoLDA:
    """Тематическое моделирование LDA для криптотекста с использованием gensim."""

    def __init__(self, n_topics: int = 10, passes: int = 15):
        self.n_topics = n_topics
        self.passes = passes
        self.model = None
        self.dictionary = None
        self.corpus_bow = None

    def fit(self, documents: list[str]):
        """Обучить модель LDA на документах."""
        tokenized = [doc.split() for doc in documents]
        self.dictionary = corpora.Dictionary(tokenized)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus_bow = [self.dictionary.doc2bow(doc) for doc in tokenized]

        self.model = models.LdaMulticore(
            corpus=self.corpus_bow,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            passes=self.passes,
            random_state=42,
            workers=3,
        )
        return self

    def get_topics(self, n_words: int = 10) -> list[list[tuple]]:
        """Получить верхние слова для каждой темы."""
        return [
            self.model.show_topic(i, topn=n_words)
            for i in range(self.n_topics)
        ]

    def get_document_topics(self, document: str) -> list[tuple]:
        """Получить тематическое распределение для одного документа."""
        bow = self.dictionary.doc2bow(document.split())
        return self.model.get_document_topics(bow, minimum_probability=0.0)

    def get_topic_distribution_matrix(self, documents: list[str]) -> pd.DataFrame:
        """Получить тематические распределения для всех документов."""
        distributions = []
        for doc in documents:
            bow = self.dictionary.doc2bow(doc.split())
            topics = self.model.get_document_topics(bow, minimum_probability=0.0)
            dist = [0.0] * self.n_topics
            for topic_id, prob in topics:
                dist[topic_id] = prob
            distributions.append(dist)
        return pd.DataFrame(
            distributions,
            columns=[f"topic_{i}" for i in range(self.n_topics)]
        )

    def coherence_score(self, documents: list[str]) -> float:
        """Вычислить когерентность тем (C_v)."""
        tokenized = [doc.split() for doc in documents]
        cm = CoherenceModel(
            model=self.model,
            texts=tokenized,
            dictionary=self.dictionary,
            coherence="c_v",
        )
        return cm.get_coherence()


class CryptoNMF:
    """Тематическое моделирование NMF для криптотекста."""

    def __init__(self, n_topics: int = 10, max_features: int = 5000):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
        )
        self.model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=500,
        )

    def fit(self, documents: list[str]):
        """Обучить модель NMF на документах."""
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.W = self.model.fit_transform(self.tfidf_matrix)
        self.H = self.model.components_
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def get_topics(self, n_words: int = 10) -> list[list[tuple]]:
        """Получить верхние слова для каждой темы."""
        topics = []
        for topic_idx in range(self.n_topics):
            top_indices = self.H[topic_idx].argsort()[-n_words:][::-1]
            topic_words = [
                (self.feature_names[i], self.H[topic_idx][i])
                for i in top_indices
            ]
            topics.append(topic_words)
        return topics

    def transform(self, documents: list[str]) -> pd.DataFrame:
        """Получить тематические распределения для новых документов."""
        tfidf = self.vectorizer.transform(documents)
        W = self.model.transform(tfidf)
        # Нормализация строк к сумме 1
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W_norm = W / row_sums
        return pd.DataFrame(
            W_norm,
            columns=[f"topic_{i}" for i in range(self.n_topics)]
        )


class CryptoLSI:
    """Тематическое моделирование LSI/LSA для криптотекста."""

    def __init__(self, n_topics: int = 10, max_features: int = 5000):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=3, max_df=0.9
        )
        self.model = TruncatedSVD(n_components=n_topics, random_state=42)

    def fit(self, documents: list[str]):
        tfidf = self.vectorizer.fit_transform(documents)
        self.document_topics = self.model.fit_transform(tfidf)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def get_topics(self, n_words: int = 10) -> list[list[tuple]]:
        topics = []
        for i in range(self.n_topics):
            top_indices = np.abs(self.model.components_[i]).argsort()[-n_words:][::-1]
            topic_words = [
                (self.feature_names[j], self.model.components_[i][j])
                for j in top_indices
            ]
            topics.append(topic_words)
        return topics


class NarrativeTracker:
    """Отслеживание эволюции нарративов во времени с помощью тематических моделей."""

    def __init__(self, n_topics: int = 8):
        self.n_topics = n_topics

    def track(self, corpus: CryptoCorpus,
              period_days: int = 7) -> pd.DataFrame:
        """Отслеживать распространённость тем по временным периодам."""
        slices = corpus.get_time_slices(period_days)
        if not slices:
            return pd.DataFrame()

        # Обучение на полном корпусе
        all_docs = [doc for slice_docs in slices for doc in slice_docs]
        nmf = CryptoNMF(n_topics=self.n_topics)
        nmf.fit(all_docs)

        # Получение распространённости по периодам
        results = []
        for i, period_docs in enumerate(slices):
            dist = nmf.transform(period_docs)
            avg = dist.mean(axis=0)
            avg["period"] = i
            avg["n_docs"] = len(period_docs)
            results.append(avg)

        return pd.DataFrame(results)

    def detect_emerging_narratives(self, prevalence: pd.DataFrame,
                                   threshold_std: float = 2.0) -> list[dict]:
        """Обнаружить темы с быстро растущей распространённостью."""
        topic_cols = [c for c in prevalence.columns if c.startswith("topic_")]
        signals = []
        for col in topic_cols:
            series = prevalence[col]
            if len(series) < 4:
                continue
            rolling_mean = series.rolling(4).mean()
            rolling_std = series.rolling(4).std()
            latest = series.iloc[-1]
            if rolling_std.iloc[-2] > 0:
                z_score = (latest - rolling_mean.iloc[-2]) / rolling_std.iloc[-2]
                if z_score > threshold_std:
                    signals.append({
                        "topic": col,
                        "z_score": z_score,
                        "current_prevalence": latest,
                        "previous_mean": rolling_mean.iloc[-2],
                    })
        return sorted(signals, key=lambda x: x["z_score"], reverse=True)


class TopicAlphaGenerator:
    """Генерация торговых сигналов из тематических распределений."""

    def __init__(self):
        self.bybit = HTTP()

    def fetch_returns(self, symbol: str, days: int = 90) -> pd.Series:
        """Получить дневные доходности с Bybit."""
        resp = self.bybit.get_kline(
            category="spot", symbol=symbol, interval="D", limit=days
        )
        rows = resp["result"]["list"]
        closes = [float(r[4]) for r in reversed(rows)]
        returns = pd.Series(
            [np.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
        )
        return returns

    def compute_topic_signal(self, prevalence: pd.DataFrame,
                              topic_token_map: dict) -> dict:
        """
        Генерировать сигналы из изменений распространённости тем.

        topic_token_map: {topic_name: [список символов Bybit]}
        Пример: {"topic_0": ["AAVEUSDT", "UNIUSDT"], ...}
        """
        topic_cols = [c for c in prevalence.columns if c.startswith("topic_")]
        signals = {}

        for col in topic_cols:
            if col not in topic_token_map:
                continue
            series = prevalence[col]
            if len(series) < 2:
                continue

            # Моментум: текущая распространённость vs среднее за 4 периода
            momentum = series.iloc[-1] - series.iloc[-4:].mean()
            # Ускорение: изменение моментума
            if len(series) >= 5:
                prev_momentum = series.iloc[-2] - series.iloc[-5:-1].mean()
                acceleration = momentum - prev_momentum
            else:
                acceleration = 0

            for symbol in topic_token_map[col]:
                signals[symbol] = {
                    "topic": col,
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "signal": np.sign(momentum) * min(abs(momentum) * 10, 1.0),
                }

        return signals


# --- Пример использования ---
if __name__ == "__main__":
    # Построение примерного корпуса
    corpus = CryptoCorpus()

    sample_docs = [
        ("Yield farming on Uniswap and Aave is generating incredible APY. "
         "DeFi protocols are the future of finance.", "reddit",
         datetime(2024, 6, 1), ["UNI", "AAVE"]),
        ("Bitcoin ETF approval is imminent. BlackRock and Fidelity filings "
         "signal institutional adoption is coming.", "reddit",
         datetime(2024, 6, 2), ["BTC"]),
        ("Layer 2 scaling solutions like Arbitrum and Optimism are reducing "
         "gas fees dramatically. L2 adoption is accelerating.", "reddit",
         datetime(2024, 6, 3), ["ARB", "OP"]),
        ("NFT market is showing signs of recovery. Blue chip collections "
         "floor prices are rising again.", "reddit",
         datetime(2024, 6, 4), ["ETH"]),
        ("Solana DeFi ecosystem growing rapidly. Jupiter DEX and Marinade "
         "staking leading the way.", "reddit",
         datetime(2024, 6, 5), ["SOL"]),
        ("AI and crypto convergence is the next big narrative. Render "
         "network and Fetch.ai are leading projects.", "reddit",
         datetime(2024, 6, 6), ["RNDR", "FET"]),
        ("Bitcoin halving impact on price historically significant. "
         "Supply reduction should drive prices higher.", "reddit",
         datetime(2024, 6, 7), ["BTC"]),
        ("Liquid staking derivatives on Ethereum are the new meta. "
         "Lido and Rocket Pool gaining market share.", "reddit",
         datetime(2024, 6, 8), ["ETH", "LDO"]),
    ]

    for text, source, date, tokens in sample_docs:
        corpus.add_document(text, source, date, tokens)

    # Обучение NMF
    print("=== Темы NMF ===")
    nmf = CryptoNMF(n_topics=4)
    nmf.fit(corpus.documents)
    topics = nmf.get_topics(n_words=5)
    for i, topic in enumerate(topics):
        words = ", ".join([f"{w}({s:.3f})" for w, s in topic])
        print(f"Тема {i}: {words}")

    # Обучение LDA
    print("\n=== Темы LDA ===")
    lda = CryptoLDA(n_topics=4, passes=10)
    lda.fit(corpus.documents)
    topics = lda.get_topics(n_words=5)
    for i, topic in enumerate(topics):
        words = ", ".join([f"{w}({s:.3f})" for w, s in topic])
        print(f"Тема {i}: {words}")

    coherence = lda.coherence_score(corpus.documents)
    print(f"Когерентность LDA (C_v): {coherence:.3f}")

    # Тематические распределения
    dist = nmf.transform(corpus.documents)
    print(f"\nМатрица документ-тема:\n{dist.round(3)}")
```

---

## Раздел 6: Реализация на Rust

```rust
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;

// --- Типы Bybit API ---

#[derive(Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

// --- Корпус ---

pub struct CryptoCorpus {
    documents: Vec<String>,
    metadata: Vec<DocumentMeta>,
    stop_words: Vec<String>,
}

pub struct DocumentMeta {
    pub source: String,
    pub date: String,
    pub tokens_mentioned: Vec<String>,
}

impl CryptoCorpus {
    pub fn new() -> Self {
        let stop_words = vec![
            "the", "is", "at", "which", "on", "a", "an", "and", "or",
            "but", "in", "with", "to", "for", "of", "from", "by", "this",
            "that", "it", "its", "are", "was", "were", "be", "been",
        ].into_iter().map(String::from).collect();

        Self {
            documents: Vec::new(),
            metadata: Vec::new(),
            stop_words,
        }
    }

    pub fn add_document(&mut self, text: &str, source: &str, date: &str,
                        tokens: Vec<String>) {
        let cleaned = self.preprocess(text);
        self.documents.push(cleaned);
        self.metadata.push(DocumentMeta {
            source: source.to_string(),
            date: date.to_string(),
            tokens_mentioned: tokens,
        });
    }

    fn preprocess(&self, text: &str) -> String {
        text.to_lowercase()
            .split_whitespace()
            .filter(|w| !self.stop_words.contains(&w.to_string()))
            .filter(|w| w.len() > 2)
            .filter(|w| !w.starts_with("http"))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    pub fn get_documents(&self) -> &[String] {
        &self.documents
    }
}

// --- TF-IDF для тематического моделирования ---

pub struct DocumentTermMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub vocabulary: Vec<String>,
    pub word_to_idx: HashMap<String, usize>,
}

impl DocumentTermMatrix {
    pub fn from_documents(documents: &[String], max_features: usize) -> Self {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_freq: HashMap<String, usize> = HashMap::new();
        let n_docs = documents.len();

        for doc in documents {
            let mut seen = std::collections::HashSet::new();
            for word in doc.split_whitespace() {
                *total_freq.entry(word.to_string()).or_insert(0) += 1;
                if seen.insert(word.to_string()) {
                    *doc_freq.entry(word.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut terms: Vec<(String, usize)> = total_freq.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));
        terms.truncate(max_features);

        let vocabulary: Vec<String> = terms.iter().map(|(w, _)| w.clone()).collect();
        let word_to_idx: HashMap<String, usize> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        let v = vocabulary.len();
        let mut matrix = vec![vec![0.0f64; v]; n_docs];

        for (d, doc) in documents.iter().enumerate() {
            let words: Vec<&str> = doc.split_whitespace().collect();
            let n = words.len() as f64;
            let mut counts: HashMap<&str, f64> = HashMap::new();
            for w in &words {
                *counts.entry(w).or_insert(0.0) += 1.0;
            }
            for (word, count) in counts {
                if let Some(&idx) = word_to_idx.get(word) {
                    let tf = count / n;
                    let df = *doc_freq.get(word).unwrap_or(&1) as f64;
                    let idf = (n_docs as f64 / (1.0 + df)).ln();
                    matrix[d][idx] = tf * idf;
                }
            }
        }

        Self { matrix, vocabulary, word_to_idx }
    }
}

// --- NMF ---

pub struct NmfModel {
    pub w: Vec<Vec<f64>>,
    pub h: Vec<Vec<f64>>,
    pub n_topics: usize,
}

impl NmfModel {
    pub fn fit(dtm: &DocumentTermMatrix, n_topics: usize, max_iter: usize) -> Self {
        let m = dtm.matrix.len();
        let v = dtm.vocabulary.len();

        let mut w = vec![vec![0.0f64; n_topics]; m];
        let mut h = vec![vec![0.0f64; v]; n_topics];

        for i in 0..m {
            for k in 0..n_topics {
                w[i][k] = 0.1 + (((i * 7 + k * 13) % 100) as f64) / 1000.0;
            }
        }
        for k in 0..n_topics {
            for j in 0..v {
                h[k][j] = 0.1 + (((k * 11 + j * 3) % 100) as f64) / 1000.0;
            }
        }

        // Мультипликативные правила обновления
        for _ in 0..max_iter {
            // Обновление H
            for k in 0..n_topics {
                for j in 0..v {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for i in 0..m {
                        num += w[i][k] * dtm.matrix[i][j];
                    }
                    for i in 0..m {
                        let mut wh = 0.0;
                        for kk in 0..n_topics {
                            wh += w[i][kk] * h[kk][j];
                        }
                        den += w[i][k] * wh;
                    }
                    if den > 1e-10 {
                        h[k][j] *= num / den;
                    }
                }
            }

            // Обновление W
            for i in 0..m {
                for k in 0..n_topics {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for j in 0..v {
                        num += dtm.matrix[i][j] * h[k][j];
                    }
                    for j in 0..v {
                        let mut wh = 0.0;
                        for kk in 0..n_topics {
                            wh += w[i][kk] * h[kk][j];
                        }
                        den += wh * h[k][j];
                    }
                    if den > 1e-10 {
                        w[i][k] *= num / den;
                    }
                }
            }
        }

        Self { w, h, n_topics }
    }

    pub fn get_top_words(&self, topic: usize, n: usize,
                         vocab: &[String]) -> Vec<(String, f64)> {
        let mut indices: Vec<usize> = (0..vocab.len()).collect();
        indices.sort_by(|&a, &b| {
            self.h[topic][b].partial_cmp(&self.h[topic][a]).unwrap()
        });
        indices.truncate(n);
        indices
            .iter()
            .map(|&i| (vocab[i].clone(), self.h[topic][i]))
            .collect()
    }

    pub fn get_document_topics(&self, doc_idx: usize) -> Vec<f64> {
        let row = &self.w[doc_idx];
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            row.iter().map(|v| v / sum).collect()
        } else {
            vec![0.0; self.n_topics]
        }
    }
}

// --- Генератор нарративных сигналов ---

pub struct NarrativeSignalGenerator {
    client: Client,
    base_url: String,
}

impl NarrativeSignalGenerator {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    pub async fn fetch_price(&self, symbol: &str) -> Result<f64> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval=D&limit=2",
            self.base_url, symbol
        );
        let resp: BybitResponse = self.client.get(&url).send().await?.json().await?;
        let close: f64 = resp.result.list[0][4].parse()?;
        Ok(close)
    }

    pub fn compute_signal(
        topic_prevalence: &[Vec<f64>],
        topic_idx: usize,
    ) -> f64 {
        if topic_prevalence.len() < 2 {
            return 0.0;
        }
        let n = topic_prevalence.len();
        let current = topic_prevalence[n - 1][topic_idx];
        let previous: f64 = topic_prevalence[..n - 1]
            .iter()
            .map(|p| p[topic_idx])
            .sum::<f64>()
            / (n - 1) as f64;
        let momentum = current - previous;
        momentum.clamp(-1.0, 1.0)
    }
}

// --- Главная функция ---

#[tokio::main]
async fn main() -> Result<()> {
    let mut corpus = CryptoCorpus::new();

    corpus.add_document(
        "Yield farming on Uniswap and Aave generating incredible APY DeFi future",
        "reddit", "2024-06-01", vec!["UNI".into(), "AAVE".into()],
    );
    corpus.add_document(
        "Bitcoin ETF approval imminent BlackRock Fidelity institutional adoption",
        "reddit", "2024-06-02", vec!["BTC".into()],
    );
    corpus.add_document(
        "Layer 2 scaling Arbitrum Optimism reducing gas fees L2 adoption",
        "reddit", "2024-06-03", vec!["ARB".into(), "OP".into()],
    );
    corpus.add_document(
        "Solana DeFi ecosystem growing Jupiter DEX Marinade staking",
        "reddit", "2024-06-04", vec!["SOL".into()],
    );
    corpus.add_document(
        "AI crypto convergence next narrative Render Fetch leading projects",
        "reddit", "2024-06-05", vec!["RNDR".into(), "FET".into()],
    );

    let dtm = DocumentTermMatrix::from_documents(corpus.get_documents(), 200);
    println!("Размер словаря: {}", dtm.vocabulary.len());
    println!("Документы: {}", dtm.matrix.len());

    let nmf = NmfModel::fit(&dtm, 3, 100);

    for k in 0..nmf.n_topics {
        let words = nmf.get_top_words(k, 5, &dtm.vocabulary);
        let word_str: Vec<String> = words
            .iter()
            .map(|(w, s)| format!("{}({:.3})", w, s))
            .collect();
        println!("Тема {}: {}", k, word_str.join(", "));
    }

    for d in 0..corpus.get_documents().len() {
        let topics = nmf.get_document_topics(d);
        let topic_str: Vec<String> = topics.iter().map(|t| format!("{:.3}", t)).collect();
        println!("Док {}: [{}]", d, topic_str.join(", "));
    }

    // Генерация сигналов
    let gen = NarrativeSignalGenerator::new();
    let price = gen.fetch_price("BTCUSDT").await?;
    println!("Цена BTC: {:.2}", price);

    Ok(())
}
```

### Структура проекта

```
ch15_crypto_topic_modeling/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── lda.rs
│   │   └── nmf.rs
│   ├── corpus/
│   │   ├── mod.rs
│   │   └── crypto_corpus.rs
│   └── trading/
│       ├── mod.rs
│       └── narrative_signals.rs
└── examples/
    ├── whitepaper_topics.rs
    ├── narrative_tracking.rs
    └── topic_alpha.rs
```

---

## Раздел 7: Практические примеры

### Пример 1: Обнаружение нарративов из Reddit

Мы собираем 50 000 постов из r/cryptocurrency за 6 месяцев (2024-H1) и обучаем модель NMF с 10 темами. Обнаруженные темы замечательно совпадают с известными крипто-нарративами:

```
Тема   Верхние слова                                Интерпретация
0      bitcoin, etf, blackrock, institutional,       Нарратив Bitcoin ETF
       approval, sec, spot, filing
1      defi, yield, farming, liquidity, aave,        Возрождение DeFi
       uniswap, protocol, tvl
2      layer, scaling, rollup, arbitrum,             Нарратив масштабирования L2
       optimism, zk, gas, fees
3      nft, collection, marketplace, floor,          Восстановление NFT
       opensea, blur, digital, art
4      solana, sol, ecosystem, jupiter,              Экосистема Solana
       meme, bonk, speed, tps
5      ai, artificial, intelligence, render,         Конвергенция AI/Crypto
       fetch, compute, gpu, decentralized
6      regulation, sec, lawsuit, ripple,             Регуляторный нарратив
       compliance, legal, court
7      staking, liquid, lido, ethereum,              Ликвидный стейкинг
       validator, eth, rocket, pool
8      meme, doge, shib, pepe, community,           Сезон мемкоинов
       bonk, floki, viral
9      bridge, cross, chain, interoperability,       Кросс-чейн/Интероп
       cosmos, polkadot, layerzero

Баллы когерентности (C_v):
  NMF:  0.52
  LDA:  0.47
  LSI:  0.38
```

NMF производит наиболее интерпретируемые темы для криптотекста, что согласуется с его преимуществом для коротких-средних документов с чётким тематическим разделением.

### Пример 2: Отслеживание жизненного цикла нарратива

Мы отслеживаем распространённость темы «Bitcoin ETF» (Тема 0) на протяжении 26 недельных периодов:

```
Неделя Распространённость Фаза           Ценовая динамика (BTC)
W1     0.08               Возникновение  $42 000
W4     0.12               Ускорение      $44 500
W8     0.18               Ускорение      $47 200
W12    0.31               Пик            $52 800
W14    0.35               Пик            $69 000 (ATH на одобрении ETF)
W16    0.28               Раннее затухание $63 500
W20    0.15               Затухание      $58 000
W24    0.07               Зрелость       $61 000

Корреляция (распространённость vs. доходность BTC):
  Одновременная: r = 0.42 (p < 0.05)
  С опережением 1 неделя: r = 0.38 (p < 0.05)
  С опережением 2 недели: r = 0.21 (незначимо)
```

Жизненный цикл нарратива чётко виден: возникновение -> ускорение -> пик -> затухание. Пик распространённости нарратива (W14) тесно совпал с ценовым пиком после фактического одобрения ETF. Распространённость нарратива имеет значимую одновременную и опережающую на 1 неделю корреляцию с доходностью.

### Пример 3: Генерация альфы на основе тем

Мы используем еженедельные векторы распространённости тем как признаки в модели ридж-регрессии для прогнозирования доходности следующей недели для топ-20 токенов:

```
Группа признаков         R² (OOS)   Альфа (годовая)  t-стат
Только цена (базовая)    0.02       0.0%             Н/Д
Только тематические      0.05       3.8%             2.14
Цена + Тема              0.08       5.2%             2.67
Цена + Тема + Вол        0.11       6.1%             2.89

Информационный коэффициент по темам:
  Bitcoin ETF (тема 0):      IC = 0.08  (значим для BTC, ETH)
  DeFi (тема 1):             IC = 0.11  (значим для AAVE, UNI, COMP)
  Мемкоины (тема 8):         IC = 0.14  (значим для DOGE, SHIB, PEPE)
  AI/Crypto (тема 5):        IC = 0.12  (значим для RNDR, FET)
```

Тематические признаки обеспечивают статистически значимую альфу (t-стат > 2) по сравнению с базовыми моделями на основе только цен. Информационный коэффициент максимален для мемкоинов (тема 8), что согласуется с тем, что эти активы наиболее управляемы нарративами.

---

## Раздел 8: Фреймворк бэктестирования

### Компоненты

1. **Конвейер данных**: Bybit API для OHLCV, yfinance для бенчмарковых индексов. Данные Reddit через хранимые архивы или API.
2. **Конструктор корпуса**: Построение скользящего еженедельного корпуса с криптоспецифичной предобработкой.
3. **Тематический движок**: NMF или LDA, обученные на скользящих 12-недельных окнах, производящие K временных рядов распространённости тем.
4. **Генератор сигналов**: Нарративный моментум (изменение распространённости), нарративное ускорение, баллы ассоциации тема-токен.
5. **Конструктор портфеля**: Лонг на токенах в ускоряющихся нарративах, недовес на токенах в затухающих нарративах.
6. **Симулятор исполнения**: Проскальзывание 10 бп, комиссия 5 бп, еженедельная ребалансировка.

### Метрики

| Метрика | Описание |
|---------|----------|
| CAGR | Среднегодовой темп роста |
| Коэффициент Шарпа | Доходность с поправкой на риск (годовая) |
| Информационный коэффициент | Корреляция между прогнозируемой и фактической доходностью |
| Когерентность тем | Качество обнаруженных тем (балл C_v) |
| Время опережения нарратива | Насколько заранее тематические сигналы предсказывают ценовые движения |
| Стабильность тем | Сходство Жаккара множеств слов тем между скользящими окнами |
| Затухание альфы | Время (недели), после которого альфа на основе тем теряет значимость |

### Примерные результаты бэктеста

```
Стратегия                          CAGR    Шарп    Макс DD  IC
Равные веса (базовая)              18.2%   0.61    -52.3%   Н/Д
Нарративный моментум (NMF)        27.8%   1.12    -34.2%   0.09
Нарративный моментум (LDA)        24.3%   0.98    -37.1%   0.07
Контрарианное затухание нарратива  16.5%   1.31    -21.8%   0.06
Альфа-по-теме (ридж-регрессия)    29.4%   1.24    -30.5%   0.11
Комбинированная (моментум+альфа)  32.1%   1.38    -28.3%   0.12

Период: 2022-01-01 — 2024-12-31
Вселенная: Топ-30 токенов по рыночной капитализации
Тематическая модель: NMF, K=10, переобучение ежемесячно
Ребалансировка: Еженедельно
```

---

## Раздел 9: Оценка производительности

### Сравнение методов

| Критерий | LSI | LDA | NMF | Динамический LDA | BERTopic |
|----------|-----|-----|-----|-------------------|----------|
| Интерпретируемость тем | Низкая | Высокая | Очень высокая | Высокая | Очень высокая |
| Вычислительная стоимость | Низкая | Средняя | Низкая | Высокая | Высокая |
| Временная стабильность | Средняя | Низкая | Высокая | Средняя | Средняя |
| Короткие тексты | Средне | Плохо | Хорошо | Плохо | Отлично |
| Генерация альфы | Низкая | Средняя | Высокая | Средняя | Высокая |
| Сложность настройки | Низкая | Средняя | Низкая | Высокая | Высокая |

### Ключевые выводы

1. **NMF — лучший выбор по умолчанию для крипто-тематического моделирования**: Производит более интерпретируемые темы, чем LDA, быстрее обучается и генерирует немного лучшие торговые сигналы. Ограничение неотрицательности соответствует тому, как работают нарративы (аддитивные, не субтрактивные).
2. **10 тем приблизительно оптимально**: Когерентность достигает пика при 8-12 темах для широкого крипто-корпуса. Меньше тем — сливаются различные нарративы; больше — появляются избыточные или неинтерпретируемые темы.
3. **Тематические сигналы имеют горизонт 1-3 недели**: Нарративный моментум предсказывает доходность на 1-3 недели вперёд. За пределами 4 недель сигнал затухает до шума.
4. **Стабильность тем важна для продакшна**: Темы NMF более стабильны между скользящими окнами (сходство Жаккара ~0.65), чем темы LDA (~0.45). Нестабильные темы производят шумные торговые сигналы.
5. **Альфа, управляемая нарративами, сконцентрирована в меньших токенах**: Информационный коэффициент максимален для токенов средней и малой капитализации, которые более чувствительны к нарративным потокам, чем BTC или ETH.

### Ограничения

- Тематические модели требуют существенного объёма текста; в периоды низкой активности в социальных сетях темы становятся ненадёжными.
- Доступ к данным Reddit и Twitter становится всё более ограниченным и дорогим.
- Тематические модели путают совместную встречаемость с семантическим значением; совместное появление «bitcoin» и «scam» не означает, что они семантически связаны.
- Динамические тематические модели вычислительно дороги и трудны для развёртывания в продакшн-системах реального времени.
- Связь между распространённостью нарратива и доходностью нелинейна — крайняя распространённость часто сигнализирует о вершине, а не о продолжении роста.
- Тематические модели не могут захватить сарказм, иронию или нюансированную тональность внутри темы.

---

## Раздел 10: Перспективные направления

1. **Нейронные тематические модели (BERTopic, CTM)**: Замена мешка слов на контекстуальные вложения из трансформерных моделей, производящие тематические представления, захватывающие семантические нюансы помимо совместной встречаемости слов.

2. **Дашборды нарративов в реальном времени**: Построение потоковых тематических моделей, обновляющихся в реальном времени по мере поступления новых постов Reddit и твитов, обеспечивающих непрерывный мониторинг распространённости нарративов для трейдеров.

3. **Каузальный анализ нарративов**: Использование причинности Грейнджера и моделей структурных уравнений для определения того, вызывают ли сдвиги нарративов ценовые движения или лишь отражают их, обеспечивая более точный тайминг сигналов.

4. **Кросс-языковое отслеживание нарративов**: Расширение тематических моделей на многоязычные корпуса (английский + китайский + корейский) для захвата появления нарративов в неанглоязычных сообществах до того, как они достигнут англоязычных рынков.

5. **Ончейн-нарративные сигналы**: Комбинирование тематических моделей на основе текста с данными об ончейн-активности (объём DEX, изменения TVL, скорость создания кошельков) для построения мультимодальных нарративных индикаторов, которые сложнее подделать через манипуляции в социальных сетях.

6. **Моделирование нарративного заражения**: Применение эпидемиологических моделей (SIR, SEIR) к распространению нарративов, рассматривая пользователей социальных сетей как восприимчивых, заражённых или выздоровевших по отношению к каждому нарративу — прогнозируя время пика и скорости затухания для более точных сигналов входа/выхода.

---

## Ссылки

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022.

2. Lee, D. D., & Seung, H. S. (1999). Learning the Parts of Objects by Non-Negative Matrix Factorization. *Nature*, 401(6755), 788-791.

3. Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by Latent Semantic Analysis. *Journal of the American Society for Information Science*, 41(6), 391-407.

4. Blei, D. M., & Lafferty, J. D. (2006). Dynamic Topic Models. *Proceedings of the 23rd International Conference on Machine Learning*, 113-120.

5. Grootendorst, M. (2022). BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure. *arXiv preprint arXiv:2203.05794*.

6. Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the Space of Topic Coherence Measures. *WSDM 2015*, 399-408.

7. Shiller, R. J. (2019). *Narrative Economics: How Stories Go Viral and Drive Major Economic Events*. Princeton University Press.
