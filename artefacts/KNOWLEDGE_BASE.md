# База знаний по проекту MTMCT PoC

Многоцелевое многокамерное отслеживание людей (MTMCT, CPU-only, FastAPI + React, метрики и эксперименты). Документ вынимает из проекта то, что переносится на **смежные области**: видеоаналитика, MOT/ReID, слияние данных нескольких сенсоров, ритейл/склад/транспорт/спорт, платформы ML-экспериментов, CPU-инференс, методика оценки, инженерия ML-продуктов, агентная разработка, деплой.

Источники внутри проекта: `artefacts/TZ_MTMCT_PoC.md` (ТЗ), `artefacts/REPORT.md` (результаты, абляции, гипотезы H1-H10), `README.md`, `agent-dump/HANDOFF.md` и `PLAN.md` (журнал решений), `agent-dump/reports/*` (T1-T13b, FIX, REVIEW), `agent-dump/inputs/research/*` (исследование моделей, датасетов, практик). Версия базы: 2026-11-03.

Область «без перекрытия камер» в проекте покрыта только в рамках `REPORT.md` и исследования `nonoverlap_datasets.md` (профиль `nonoverlap`, CHIRLA, гипотеза H3 отвергнута); отдельное развитие этого направления владелец отменил, поэтому здесь оно описано лишь как накопленный опыт и справка.

---

## 1. Как пользоваться

### 1.1 Карта разделов

| Нужно | Раздел |
|---|---|
| Понять, на какие задачи это похоже | 2 |
| Выбрать архитектуру под свои камеры | 3 |
| Выбрать модель/библиотеку на стадию, проверить лицензию и CPU-пригодность | 4 |
| Правильно измерить качество | 5 |
| Найти данные для проверки | 6 |
| Собрать платформу экспериментов, ускорить CPU-инференс | 7 |
| Развернуть на сервере | 8 |
| Организовать работу агентов Claude | 9 |
| Быстро найти симптом ошибки | 10 |
| Не нарушить закон и лицензии | 11 |
| Куда двигаться дальше | 12 |

### 1.2 Легенда ссылок

- **RU** - русскоязычный источник (Хабр, документация на русском, ODS, vc.ru). **EN** - английский аналог (arXiv, официальная документация, GitHub, блог).
- **▶RU / ▶EN** - видео YouTube (русское / английское). Все видео проверены через oEmbed YouTube (существуют, название и канал совпадают). Если русского видео нет, стоит пометка «▶RU: нет».
- «Не проверено» - ссылки, которые автоматическая проверка не смогла открыть (сайт отдаёт 403/302 боту), но они присутствовали в выдаче поиска или в исследовании проекта: docs.opencv.org (гомография), mdpi.com и sciencedirect.com (обзоры MTMCT), zenodo.org (CITIC-MTMC), law.ru (редирект), eur-lex.europa.eu (отдаёт 202). Остальные 250+ ссылок вернули 200.
- Ссылки проверены 2026-11-03 запросом HTTP (код 200) либо присутствовали в выдаче поиска; arXiv-страницы сверены по названию статьи.
- Лицензионные пометки: **perm** - Apache-2.0/MIT/BSD, **copyleft** - GPL/AGPL, **NC** - только некоммерческое использование.

### 1.3 Главные выводы проекта (если читать только это)

1. **Узкое место - трекинг внутри камеры, а не межкамерная склейка.** Oracle-режим (идеальные треклеты из разметки) даёт межкамерный IDF1 0.97-1.00 на перекрывающихся наборах, полный путь из видео даёт 0.37-0.76. Сначала чинить детекцию и треклеты, потом ассоциацию.
2. **Кеш тяжёлых стадий + дешёвая ассоциация** превращают подбор порогов и весов из часов в секунды (2-10 с вместо 12-19 мин на прогон).
3. **Разрез треклетов по скачку внешности** - главный рычаг качества (+0.16 межкамерного IDF1, ложные слияния 19 → 8 на garden1).
4. **Геометрия (гомография, точка ног) помогает только на чистых треклетах**; при шумной калибровке жёсткий gate вредит. Мягкая стоимость и медиана расстояния безопаснее.
5. **Пороги привязаны к шкале косинуса конкретной ReID-модели** - при смене модели пересобирать.
6. **Tune/test-протокол и контрольные абляции** нужны, потому что знаки эффекта на tune и test расходятся (геометрия на AIC24: +0.03 и -0.07).
7. **Лицензии и право** - отдельная работа: ultralytics/boxmot (AGPL), insightface (NC), биометрия (152-ФЗ, GDPR ст. 9, AI Act).

---

## 2. Постановки задач и где они встречаются

| Задача | Суть | Вход → выход | Смежные области | Где в проекте |
|---|---|---|---|---|
| **MOT** (однокамерный трекинг) | Детекции в кадрах → траектории с ID | видео → треки | подсчёт людей, спорт, робототехника, склад | стадии detect, track |
| **ReID** (повторная идентификация) | Узнать тот же объект по внешности на другом кадре/камере | кроп → вектор (эмбеддинг) | поиск по галерее, ритейл-аналитика, транспорт (авто) | стадия reid, портрет треклета |
| **MTMCT** | Одна сущность = один глобальный ID во всех камерах | N видео → глобальные треки | охрана объекта, ритейл, аэропорты, умный город | весь пайплайн |
| **Многоракурсное обнаружение** (multi-view detection) | Слить виды в одно пространство (пол/BEV) и детектировать там | N кадров + калибровка → позиции на плане | толпа, стадионы, склад | WILDTRACK, геометрия |
| **Cross-camera analytics** | Время пребывания, пути, тепловые карты, очереди | глобальные треки → отчёты | ритейл, склад, транспорт | выходные таблицы `global_tracks.csv` |
| **Слияние сенсоров** | Те же принципы ассоциации для камер + радар/лидар/RFID | наблюдения → единый объект | беспилотники, умные склады | идея переносится: стоимость = сумма сигналов, ограничения cannot-link |
| **Открытый набор (open-set)** | Нет заранее известной галереи | — | промышленные площадки | MICRO-TRACK |
| **Не перекрывающиеся камеры** | Между камерами слепые зоны | треклеты + топология | коридоры, здания | профиль `nonoverlap` (описан в REPORT/исследовании) |

Где задача встречается в смежных областях и что берём из проекта:

| Область | Чем похоже | Что переносить | Что другое |
|---|---|---|---|
| Ритейл | Покупатели между зонами/камерами, время в зале | каскад с поздним слиянием, зоны-события, ReID без лица | потолочные fisheye-камеры, много одинаковой одежды у персонала |
| Склад/логистика | Люди и техника по цехам | гомография на пол, ограничения cannot-link в один кадр | плотные стеллажи (окклюзии), униформа |
| Транспорт/умный город | Те же методы, но объект - авто | ReID + топология дорог + время проезда (CityFlow) | калибровка и 3D, большие масштабы |
| Спорт | Игроки в нескольких ракурсах | гомография на поле, ReID по номеру/форме, ByteTrack | похожая форма, быстрые движения |
| Беспилотники/робототехника | Многокамерный BEV | BEV-слияние, стоимость движения | реальное время, GPU |
| ML-платформы | Эксперименты с кешем и сравнением | контракты таблиц, реестр, контент-адресный кеш, sweep | — |

Ссылки на постановки:

| Тема | RU | EN | Видео |
|---|---|---|---|
| Обзоры MTMCT / MOT | [Самая сложная задача в CV (Хабр)](https://habr.com/ru/companies/recognitor/articles/505694/); [Многокамерная видеоаналитика (Хабр)](https://habr.com/ru/articles/117746/) | [MOT: A Literature Review](https://arxiv.org/abs/1409.7618); [In Pursuit of Many (обзор MOT)](https://arxiv.org/abs/2209.04796); [Multi-camera multi-object tracking: a review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0925231223006811); [MOMCT для транспорта (MDPI)](https://www.mdpi.com/1424-8220/23/8/3852) | ▶RU: нет обзора MTMCT; ▶EN [5th BMTT MOTChallenge Workshop](https://www.youtube.com/watch?v=NVc0jll-Lcw) |
| Ритейл-аналитика | [Трекинг посетителей на fisheye-камерах (Хабр)](https://habr.com/ru/articles/1039354/); [Видеоаналитика на open source до продакшена (Хабр)](https://habr.com/ru/articles/981268/) | [retail_video_analytics (GitHub)](https://github.com/kellinpelrine/retail_video_analytics) | ▶RU: нет; ▶EN [Tracking objects across multiple cameras with Metropolis](https://www.youtube.com/watch?v=oBL84CeLwKk) |
| Спорт | [Трекинг для командных видов спорта (Хабр)](https://habr.com/ru/articles/774298/) | [DanceTrack: MOT с одинаковой внешностью](https://arxiv.org/abs/2111.14690) | ▶EN [ByteTrack на спорте](https://www.youtube.com/watch?v=6LGpf-a1K1Q) |
| Транспорт | [Пайплайн распознавания ТС (Хабр)](https://habr.com/ru/articles/838098/) | [CityFlow](https://arxiv.org/abs/1903.09254); [8th AI City Challenge](https://arxiv.org/abs/2404.09432) | ▶EN [AIC 2019 Track 1 MTMC demo](https://www.youtube.com/watch?v=WS2Q6yhMxdI) |
| Промышленность, открытый набор | [Криптонит: реидентификация 2.0 (Хабр)](https://habr.com/ru/companies/kryptonite/articles/704914/) | [MICRO-TRACK / Facility-ReID](https://arxiv.org/abs/2409.03879) | ▶EN [ReID: введение (Pysource)](https://www.youtube.com/watch?v=SMRLT-jbwgo) |

---

## 3. Архитектурные подходы

### 3.1 Эволюция подходов (по схеме владельца)

| Эпоха | Подход | Суть | Плюсы | Минусы | Ссылки |
|---|---|---|---|---|---|
| 2020 → | **Ветка CityFlow** (авто): ReID + однокамерные треки + пространственно-временная ассоциация | Каскад, поздняя склейка треклетов | Модульно, объяснимо, CPU-пригодно | Ошибка накапливается по стадиям | EN [CityFlow](https://arxiv.org/abs/1903.09254) |
| 2021-2023 | **Раннее слияние / BEV** | Проекция признаков всех камер на пол/BEV, детекция и трекинг в общем пространстве | Лучшая точность при перекрытии и точной калибровке (WILDTRACK IDF1 ~95) | Нужна точная калибровка, обучение, GPU | EN [MVDet](https://arxiv.org/abs/2007.07247); [EarlyBird (GitHub)](https://github.com/tteepe/EarlyBird); [TrackTacular](https://arxiv.org/abs/2403.12573); [BEVFusion](https://arxiv.org/abs/2205.13542) |
| 2022-2024 | **Tracklet + граф (позднее слияние)** | Узлы - треклеты, рёбра - близость, решатель кластеризации/графа | Масштабируется, лучший в AI City (AIC23/24) | Зависит от качества треклетов | EN [Anchor-guided clustering (AIC23)](https://arxiv.org/abs/2304.09471); [ReST](https://arxiv.org/abs/2308.13229); [MPNTrack](https://arxiv.org/abs/1912.07515); [UMPN](https://arxiv.org/abs/2507.08494) |
| 2024-2026 | **Сквозные модели** (MOTR, MCTR, MOTIP) | Трекинг как предсказание ID трансформером | Нет эвристик ассоциации | Нужно обучение, данные, GPU | EN [MOTR](https://arxiv.org/abs/2105.03247); [MOTIP](https://arxiv.org/abs/2403.16848); [MCTR](https://openaccess.thecvf.com/content/WACV2025W/RWS/papers/Niculescu-Mizil_MCTR_Multi_Camera_Tracking_Transformer_WACVW_2025_paper.pdf) |

Онлайн против офлайн: по AIC24 лучшая офлайн-склейка 71.9 HOTA против ~67.2 у лучшей онлайн (из исследования проекта); офлайн побеждает, пока нет требования реального времени. Для нашего PoC (офлайн, CPU) выбран **офлайн-каскад «A»**. Гипотеза H10 («онлайн-решатель уступает офлайну») в проекте не проверялась.

| Тема | RU | EN | Видео |
|---|---|---|---|
| Каскад и tracking-by-detection | [Object Tracking на YOLO и DeepSORT (Хабр)](https://habr.com/ru/articles/514450/); [Трекинг без разметки (ODS на Хабре)](https://habr.com/ru/companies/ods/articles/705020/) | [SORT vs DeepSORT vs ByteTrack vs OC-SORT (Roboflow)](https://roboflow.com/blog/sort-vs-deepsort-vs-bytetrack-vs-oc-sort) | ▶RU [Реализация SORT & DeepSORT](https://www.youtube.com/watch?v=Y9CBkdhGwqI); ▶EN [Сравнение ByteTrack/BoT-SORT/OC-SORT](https://www.youtube.com/watch?v=xuaippO-ShQ) |
| BEV / раннее слияние | [Методы распознавания 3D для беспилотников, Яндекс (Хабр)](https://habr.com/ru/articles/437674/); [Эволюция машинного зрения автономного транспорта, Яндекс (Хабр)](https://habr.com/ru/companies/yandex/articles/844126/) | [BEVFusion](https://arxiv.org/abs/2205.13542); [MVDet](https://arxiv.org/abs/2007.07247) | ▶EN [Как камеры становятся BEV](https://www.youtube.com/watch?v=jtStdZC5QoE); ▶EN [Efficient 3D Perception, MIT](https://www.youtube.com/watch?v=5v9wa-UCFMM) |
| Сквозные модели | RU: нет (обзор на русском не найден) | [DETR](https://arxiv.org/abs/2005.12872); [MOTRv2](https://arxiv.org/abs/2211.09791); [MeMOTR](https://arxiv.org/abs/2307.15700) | ▶RU: нет; ▶EN [MOTR (Mak Gaiduk)](https://www.youtube.com/watch?v=jK6xsNbnE-8); [MOTRv2](https://www.youtube.com/watch?v=2ciLMuyW2Qg); [DETR (Kilcher)](https://www.youtube.com/watch?v=T35ba_VXkMY) |
| Граф/GNN-ассоциация | RU: нет (на Хабре только общие статьи о GNN, например [Что такое графовые нейронные сети](https://habr.com/ru/articles/557280/)) | [Learning a Neural Solver for MOT](https://arxiv.org/abs/1912.07515); [UMPN](https://arxiv.org/abs/2507.08494); [ReST](https://arxiv.org/abs/2308.13229) | ▶RU: нет; ▶EN [Graph Networks for MOT](https://www.youtube.com/watch?v=k4bvlETA3QU) |
| Решения AI City (код победителей) | RU: нет | [AIC23 UWIPL/ETRI (GitHub)](https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI); [Код топ-команд 2024](https://github.com/NVIDIAAICITYCHALLENGE/2024AICITY_Code_From_Top_Teams); [Победители 2024](https://www.aicitychallenge.org/2024-challenge-winners/); [Победители 2025](https://www.aicitychallenge.org/2025-challenge-winners/) | ▶EN [AIC 2023 Track 1 demo](https://www.youtube.com/watch?v=0oLuCO-_ZNA); [Multi-Camera People Tracking for AIC (Excel@FIT)](https://www.youtube.com/watch?v=8Xp-TS_GpaA) |

### 3.2 Дерево решений проекта

```
Видео синхронизированы между камерами?
├─ нет → сначала синхронизация (общие часы, сдвиг frame_offset, по звуку/событию).
│        Без неё геометрия и окна времени бесполезны; остаётся только внешний вид.
└─ да → Камеры перекрываются (видят общую зону)?
   ├─ да (overlap) → Есть калибровка (гомография на пол)?
   │   ├─ да, точная (<1 м) → геометрия на плоскость пола; жёсткий gate допустим
   │   │       (EPFL Terrace). Для максимума точности и при наличии данных/GPU -
   │   │       BEV/раннее слияние (MVDet, EarlyBird, TrackTacular).
   │   ├─ да, шумная (CAMPUS: медиана 1-1.7 м, p90 до 9 м) → геометрия МЯГКАЯ:
   │   │       медиана расстояния, малый вес (0.25-0.5), широкий gate (5 м).
   │   └─ нет → только внешний вид (ReID) + cannot-link «два треклета в одном
   │            кадре одной камеры - разные люди»; при >=2 камер искать калибровку
   │            по общим точкам.
   ├─ частично (CHIRLA: камеры 1-4 смотрят в одну комнату) → модель связей камер
   │   «видны одновременно» = нет информации; сегменты-визиты, overlap=ignore
   │   (иначе окна времени дают ложные cannot-link).
   └─ нет (nonoverlap) → внешний вид + окна времени переходов между камерами
       (топология). Применять, только если в данных есть реальные переходы через
       слепую зону; на CHIRLA окна вредили (H3 отвергнута). Реальных публичных
       наборов мало: DukeMTMC снят, NLPR_MCT офлайн, MTMMC требует соглашения.
```

Что даёт каждый шаг дерева на наших данных: см. REPORT §3.1 (выбор геометрии на tune) и §5.2 (абляции стоимостей).

### 3.3 Критерии выбора «каскад или сквозная модель»

| Условие | Выбор |
|---|---|
| CPU-only, нет обучения, нужна объяснимость и смена компонент | Каскад (наш выбор) |
| Есть GPU, размеченный домен, перекрытие, точная калибровка | BEV/раннее слияние |
| Есть GPU и данные для дообучения, нужен online | Сквозные (MOTIP/MCTR) - изучать, не внедрять в PoC |
| Нужно сравнить гипотезы быстро | Каскад + кеш + oracle-режим |

---

## 4. Компоненты и технологии по стадиям

Цепочка проекта: `detect → track → reid → refine → portrait ∥ geometry → associate → postprocess → eval`. Колонка «CPU» - пригодность на ноутбуке i5-12450H (16 ГБ); «Как использовали» - факты из REPORT.

### 4.1 Детекторы людей

| Что | Лицензия | CPU | Как использовали / заметки | RU | EN |
|---|---|---|---|---|---|
| **YOLOX-s/tiny** (выбран) | Apache-2.0 (perm) | хорошо через OpenVINO/ORT: s ~102 мс/кадр, tiny@640 ~84 мс | Детекция доминирует по времени полного прогона; YOLOX-tiny ускорил стадию лишь на 12 % (декодирование видео и предобработка), IDF1 +0.02 | [Обучение YOLOX на своих классах (Хабр)](https://habr.com/ru/articles/680116/); [YOLOX vs RT-DETRv2 (Ultralytics, рус.)](https://docs.ultralytics.com/ru/compare/rtdetr-vs-yolox) | [YOLOX paper](https://arxiv.org/abs/2107.08430); [GitHub](https://github.com/Megvii-BaseDetection/YOLOX) |
| RT-DETR | Apache-2.0 в репо Baidu; **AGPL при запуске через ultralytics** | тяжелее | Не использовали; без NMS | [Выбор детектора: YOLO, RT-DETR, RF-DETR, D-FINE (AI Рассвет)](https://airassvet.ru/articles/yolo-rt-detr-rf-detr-d-fine-sravnenie) | [RT-DETR](https://arxiv.org/abs/2304.08069); [GitHub](https://github.com/lyuwenyu/RT-DETR) |
| D-FINE | Apache-2.0 | N-версия лёгкая | Планировали, не подключён | см. выше | [D-FINE](https://arxiv.org/abs/2410.13842); [GitHub](https://github.com/Peterande/D-FINE) |
| RF-DETR | Apache-2.0 (N-L), PML-1.0 (XL, 2XL) | средне | Не использовали | [Хабр: YOLOv11 и RF-DETR в видео](https://habr.com/ru/articles/1018062/) | [GitHub](https://github.com/roboflow/rf-detr) |
| YOLO (ultralytics: v8, 11, 26) | **AGPL-3.0** / платная | хорошо | Только опциональный адаптер: сетевой сервис = раскрытие исходников | [Анализ производительности YOLOv8 + OpenVINO (Хабр)](https://habr.com/ru/articles/822917/) | [Лицензии моделей детекции](https://robotinaction.tech/model-licenses.html); [ultralytics](https://github.com/ultralytics/ultralytics) |

- **Когда использовать:** ритейл/склад - YOLOX/RT-DETR/D-FINE для коммерческого продукта; YOLO-ultralytics для исследований.
- **Подводные камни:** `input_size` YOLOX должен быть кратен 32 (иначе ломается постпроцесс); на CPU дорого декодирование видео, а не сама сеть; WILDTRACK GT - проекция 3D-цилиндра, поэтому recall детектора при IoU 0.5 всего 0.58 (precision 0.48).
- Видео: ▶RU: нет; ▶EN [YOLOX paper explanation (LearnOpenCV)](https://www.youtube.com/watch?v=BwIxFkT-3FA).

### 4.2 Трекеры внутри камеры (SCT/MOT)

| Что | Идея | Лицензия | Как использовали | RU | EN |
|---|---|---|---|---|---|
| SORT | Калман + венгерский алгоритм по IoU | GPL-3.0 (репо Bewley) | Базовая идея | [Калман в картинках (Хабр)](https://habr.com/ru/articles/594249/); [Венгерский алгоритм (Хабр)](https://habr.com/ru/articles/422009/) | [SORT](https://arxiv.org/abs/1602.00763) |
| DeepSORT | + внешность и каскад по давности | GPL-3.0 | Идея ReID в треках | [Как работает Object Tracking на YOLO и DeepSORT (Хабр)](https://habr.com/ru/articles/514450/) | [DeepSORT](https://arxiv.org/abs/1703.07402) |
| **ByteTrack** (выбран) | Вторая ассоциация по низким скорам | MIT | Трекер по умолчанию; быстрый, без ReID | [Хабр: ByteTrack на GPU и мультикамерный трекинг](https://habr.com/ru/articles/1059236/) | [ByteTrack](https://arxiv.org/abs/2110.06864); [GitHub](https://github.com/FoundationVision/ByteTrack) |
| BoT-SORT | + компенсация движения камеры, ReID | MIT | Зарегистрирован, запасной | [Хабр: Кэширование и трекинг YOLO](https://habr.com/ru/articles/1033870/) | [BoT-SORT](https://arxiv.org/abs/2206.14651); [GitHub](https://github.com/NirAharon/BoT-SORT) |
| OC-SORT | Обновление по наблюдению | MIT | Зарегистрирован; хорош при нелинейном движении | [Трекинг на fisheye (OC-SORT + ReID, Хабр)](https://habr.com/ru/articles/1039354/) | [OC-SORT](https://arxiv.org/abs/2203.14360); [GitHub](https://github.com/noahcao/OC_SORT) |
| CAMELTrack | Обученная ассоциация | см. репо | Не использовали | RU: нет | [CAMELTrack](https://arxiv.org/abs/2505.01257); [GitHub](https://github.com/TrackingLaboratory/CAMELTrack) |
| BoxMOT | Набор трекеров | **AGPL-3.0** | Только опционально; альтернатива - пакет `trackers` (Apache) | [Fisheye-трекинг на BoxMOT (Хабр)](https://habr.com/ru/articles/1039354/) | [BoxMOT docs](https://mikel-brostrom.github.io/boxmot/); [roboflow/trackers](https://github.com/roboflow/trackers) |

- **Как использовали:** трекеры подключены через `mtmct/track/trackers_lib.py` (bytetrack, botsort, ocsort). Итог: треклеты фрагментированы (~10 на человека на камеру), DetA 0.5-0.67 - это и есть узкое место всей системы.
- **Подводные камни:** при дробном шаге кадров (25 → 10 к/с) интерполяция треклетов создавала рамки на необработанных кадрах (ревью M1, исправлено); `trackers` тянет `opencv-python`, конфликтующий с `-headless` (раздел 10).
- Видео: ▶RU [Реализация SORT & DeepSORT](https://www.youtube.com/watch?v=Y9CBkdhGwqI); ▶RU [Фильтр Калмана на пальцах](https://www.youtube.com/watch?v=CzSmMk8OS0c); ▶RU [Венгерский алгоритм](https://www.youtube.com/watch?v=vPPYhkAwOCk); ▶EN [ByteTrack: A Smarter Way (AIPapersAndConcepts)](https://www.youtube.com/watch?v=n3OpvFKasu0); ▶EN [Visually Explained: Kalman Filters](https://www.youtube.com/watch?v=IFeCIbljreY); ▶EN [Сравнение трекеров (Pysource)](https://www.youtube.com/watch?v=xuaippO-ShQ).

### 4.3 ReID (кросс-камерный внешний вид)

| Модель | Лицензия (код / веса) | CPU на кроп | Результат в проекте | RU | EN |
|---|---|---|---|---|---|
| **OSNet-AIN x1_0** (по умолчанию) | MIT / веса на открытых наборах | 11.0 мс (OpenVINO) | Базовый; хорошо обобщается между доменами | [Person re-identification (НКБтех, ▶)](https://www.youtube.com/watch?v=2-VrfYvrDvU) | [OSNet](https://arxiv.org/abs/1905.00953); [Torchreid](https://github.com/KaiyangZhou/deep-person-reid); [docs](https://kaiyangzhou.github.io/deep-person-reid/) |
| OSNet x0_25 | то же | 2.1 мс | IDF1 -0.002 при ускорении ReID в 2.3 раза (H8) | — | то же |
| **Intel OMZ person-reidentification-retail-0287** | Apache-2.0 | 5.7 мс | На garden2 быстрее и лучше (+0.08 IDF1), но 34 ID на 18 людей: пороги подбирать отдельно | [Хабр: Реидентификация 2.0, Криптонит](https://habr.com/ru/companies/kryptonite/articles/704914/) | [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) |
| Bag of Tricks (сильный бейзлайн) | MIT | — | Источник приёмов (BN-neck, label smoothing) | [Хабр: ReID для ТС](https://habr.com/ru/articles/838098/) | [Bag of Tricks](https://arxiv.org/abs/1903.07071); [Triplet loss](https://arxiv.org/abs/1703.07737) |
| TransReID, CLIP-ReID | MIT/исследовательские | 150-400 мс | Не использованы; верхняя граница офлайн | RU: нет | [TransReID](https://arxiv.org/abs/2102.04378); [CLIP-ReID](https://arxiv.org/abs/2211.13977) |
| FastReID | Apache-2.0 | — | Набор для дообучения | RU: нет | [fast-reid](https://github.com/JDAI-CV/fast-reid) |

- **Портрет треклета:** top-K лучших кадров (K=10) против среднего по всем рамкам. H4 не подтверждена: на garden2 среднее по всем лучше (0.643 против 0.586), на CHIRLA хуже (0.713 против 0.759), а время ReID в 3-4 раза больше.
- **Обучающие датасеты ReID** (Market-1501, MSMT17, Duke) имеют исследовательские лицензии: веса MIT, а данные не для коммерции - проверить перед продуктом. DukeMTMC снят автором.
- **Обзор и приёмы:** [Deep Learning for Person Re-identification: Survey](https://arxiv.org/abs/2001.04193); [MSMT17 / PTGAN](https://arxiv.org/abs/1711.08565).
- **Re-ranking (k-reciprocal):** [Re-ranking with k-reciprocal encoding](https://arxiv.org/abs/1701.08398). RU: нет. В проекте H6 не подтверждена: pairwise F1 вырос (0.675 → 0.730), но число ID выросло с 18 до 37, IDF1 +0.004; re-ranking меняет шкалу расстояний - порог нужно подбирать отдельно.
- Видео: ▶RU [Делаем трекинг с ReID на уровне SOTA (ODS, Козак)](https://www.youtube.com/watch?v=jXpRfAi9rEY); ▶RU [Person re-identification (НКБтех)](https://www.youtube.com/watch?v=2-VrfYvrDvU); ▶RU [Person Re-identification. Try to capture me (ODS)](https://www.youtube.com/watch?v=XuWp4OkwYvY); ▶RU [Person re-identification (семинар)](https://www.youtube.com/watch?v=O8qtBYeOSKE); ▶EN [Deep Person Re-identification Introduction](https://www.youtube.com/watch?v=BPNQqHpLSGk); ▶EN [How ReID works (Pysource)](https://www.youtube.com/watch?v=SMRLT-jbwgo).

### 4.4 Поза, точка ног, геометрия, гомография, BEV

| Что | Зачем | Лицензия | Как использовали | RU | EN |
|---|---|---|---|---|---|
| Точка ног = низ рамки (`foot_point/bottom_center`) | Координата человека на полу | — | Шум точки ног ломает геометрию на full; H9 (поза точнее низа рамки) не проверена | [Из фото в 3D, ч.2: калибровка камеры (Хабр)](https://habr.com/ru/articles/772534/); [Модель камеры (Хабр)](https://habr.com/ru/articles/709378/) | [RTMPose](https://arxiv.org/abs/2303.07399); [ViTPose](https://arxiv.org/abs/2204.12484); [mmpose](https://github.com/open-mmlab/mmpose) |
| Гомография изображение → пол | Сравнение положений между камерами в метрах | — | `geometry/homography.py`; расстояние - медиана по паре треклетов, gate 1.5 м (чистая калибровка) - 5 м (шумная) | [Эволюция машинного зрения, Яндекс: гомография для вида сверху (Хабр)](https://habr.com/ru/companies/yandex/articles/844126/) | [OpenCV: homography tutorial](https://docs.opencv.org/doc/doxygen/html/d9/dab/tutorial_homography.html) |
| BEV / раннее слияние | Единое пространство для детекции | Apache/MIT | Не реализовано; вид сверху есть в плеере для откалиброванных наборов | [Методы 3D для беспилотников, Яндекс (Хабр)](https://habr.com/ru/articles/437674/) | [MVDet](https://arxiv.org/abs/2007.07247); [TrackTacular](https://github.com/tteepe/TrackTacular); [UCMCTrack](https://arxiv.org/abs/2312.08952) |

- **Правило:** геометрия полезна только на чистых и длинных треклетах (oracle WILDTRACK 0.707 → 0.981 с геометрией). На full-режиме мягкая геометрия: tune garden1 +0.015, AIC24 +0.026, WILDTRACK -0.012; на test везде хуже (terrace -0.009, garden2 -0.028, WILDTRACK -0.033, AIC24 -0.066).
- Видео: ▶RU: нет (гомография - общих уроков на русском много, но специфики MTMCT нет); ▶EN [Computing Homography (First Principles of CV)](https://www.youtube.com/watch?v=l_qjO4cM74o); ▶EN [Bird's-Eye View with homography (Ultralytics)](https://www.youtube.com/watch?v=6_MHChha8Ds); ▶EN [OpenCV Perspective Transform](https://www.youtube.com/watch?v=jtvG8-Ikaws); ▶EN [Camera Calibration Explained](https://www.youtube.com/watch?v=Wcnb197g2i0); ▶EN [OpenCV Camera Calibration](https://www.youtube.com/watch?v=H5qbRTikxI4).

### 4.5 Межкамерная ассоциация и решатели

| Компонент | Суть | Результат в проекте | RU | EN |
|---|---|---|---|---|
| Матрица стоимостей: взвешенное среднее + **cannot-link** | Сумма сигналов (внешность, геометрия, время), жёсткие запреты «один кадр = разные люди» | Стоимость с весом 0 должна отключать и свои cannot-link (ревью H2) | [Венгерский алгоритм в слежении за объектами (Хабр)](https://habr.com/ru/articles/267651/) | [scipy linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) |
| **Агломеративная кластеризация** с порогом (выбрана) | Склейка треклетов снизу вверх по среднему расстоянию с запретами | Лучший решатель на tune и test | [Разработка и интерпретация иерархической кластеризации (Хабр)](https://habr.com/ru/articles/782862/) | [scipy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) |
| Anchor-guided (AIC23) | Якорные треклеты, остальные присоединяются | H7 не подтверждена: 0.517 против 0.586 | RU: нет | [Anchor-guided clustering](https://arxiv.org/abs/2304.09471) |
| GNN / correlation clustering | Обученная ассоциация | Не использовались | RU: нет | [MPNTrack](https://arxiv.org/abs/1912.07515); [ReST](https://arxiv.org/abs/2308.13229) |
| Топология камер и окна времени | Вероятное время перехода между камерами | H3 отвергнута на CHIRLA; модель связей: одновременно видимые пары = «нет информации» | RU: нет | [OCMCTrack (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Specker_OCMCTrack_Online_Multi-Target_Multi-Camera_Tracking_with_Corrective_Matching_Cascade_CVPRW_2024_paper.pdf) |
| Чистка треклетов (refine): **разрез по скачку внешности** | Разрезать треклет там, где эмбеддинг резко меняется (`split_thr 0.75`, окно 7) | **+0.16 межкамерного IDF1**, слияния 19 → 8 (garden1); без refine garden2 0.486 вместо 0.586 | RU: нет | идея из ReID-first подходов: [ReidTrack (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Specker_ReidTrack_Reid-Only_Multi-Target_Multi-Camera_Tracking_CVPRW_2023_paper.pdf) |

- Видео: ▶RU [Агломеративная иерархическая кластеризация, dendrogram (selfedu)](https://www.youtube.com/watch?v=JcniBqle4kw); ▶EN [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo); ▶EN [Tracking with Graph Neural Networks](https://www.youtube.com/watch?v=jnDbOlP5Qt0).
- **Пороги** зависят от шкалы косинуса ReID-модели (x0_25 и OMZ-0287 потребовали своих порогов refine и решателя): хранить таблицу «модель → пороги».

### 4.6 Готовые продукты и референсы мультикамерного трекинга

| Что | Назначение | RU | EN |
|---|---|---|---|
| NVIDIA Metropolis / DeepStream MTMC | Промышленный референс (3D, склад) | — | [MTMC App (docs)](https://docs.nvidia.com/mms/text/MDX_Multi_Camera_Tracking_App.html); [Блог NVIDIA](https://developer.nvidia.com/blog/optimize-processes-for-large-spaces-with-the-multi-camera-tracking-workflow/) |
| Frigate (распознавание лиц в NVR) | Пример, как лицо отделено от трекинга | — | [Docs](https://docs.frigate.video/configuration/face_recognition/) |
| nolanzzz/mtmct, MCT-репозитории | Дипломные реализации, оценка MTA | — | [nolanzzz/mtmct](https://github.com/nolanzzz/mtmct); [multi-camera-person-tracking](https://github.com/Abdirayimov/multi-camera-person-tracking) |
| Supervision (Roboflow) | Вспомогательные CV-инструменты | — | [docs](https://supervision.roboflow.com/); [GitHub](https://github.com/roboflow/supervision) |
| NtechLab FFmulti | Российский продукт мультикамерного трекинга | [Документация FindFace Multi](https://docs.ntechlab.com/projects/ffmulti/ru/1.1/settings.html) | [EN docs](https://docs.ntechlab.com/projects/ffmulti/en/2.4.1/video-config.html) |

---

## 5. Оценка качества

### 5.1 Метрики и что каждая показывает

| Метрика | Что измеряет | Смещение | Когда использовать | RU | EN |
|---|---|---|---|---|---|
| **MOTA / MOTP** (CLEAR MOT) | Пропуски + ложные + смены ID на кадре; локализация | В сторону детекции | Детектор + трекер, сравнение с литературой | [Забудьте про точность, десятки метрик (Tevian, Хабр)](https://habr.com/ru/companies/tevian/articles/963802/) | [Understanding MOT metrics](https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics) |
| **IDF1** (IDP/IDR) | Доля времени с правильным ID, глобальное сопоставление траекторий | В сторону ассоциации | Основная метрика MTMCT; AI City | [ML Atlas: метрики трекинга](https://ml-atlas.ru/topics/metrics/tracking-metrics) | [Performance Measures for MTMC (Ristani 2016)](https://arxiv.org/abs/1609.01775) |
| **HOTA** (DetA, AssA) | Баланс детекции, ассоциации, локализации по 19 порогам IoU | Сбалансирована | Диагностика: что чинить, детекцию или ассоциацию | см. Tevian | [HOTA (arXiv)](https://arxiv.org/abs/2009.07736); [блог автора](https://autonomousvision.github.io/hota-metrics/) |
| **TrackEval** (инструмент) | Эталонная реализация HOTA/CLEAR/Identity | — | Все метрики проекта | RU: нет | [GitHub](https://github.com/JonathonLuiten/TrackEval) |

- Видео: ▶RU: нет отдельного русского разбора метрик; ▶EN [MOT Metrics - MOTA, IDF1, HOTA: алгоритм и разбор кода (Mak Gaiduk)](https://www.youtube.com/watch?v=rDT0gkQpqYk).
- **Подводный камень TrackEval:** старый код использует `np.float`, `np.int` и т.п. - с numpy 2 падает; в проекте вендорен `third_party/TrackEval` с патчем. Сайт MOTChallenge в 2026 отдаёт «offline» (410) - данные и страницы лучше сохранять локально.

### 5.2 Многокамерная склейка в TrackEval (трюк проекта)

Камеры **склеиваются по времени в одну последовательность** (камера 2 сдвигается после камеры 1 и т.д.): один идентификатор человека в разных камерах становится одной траекторией, и HOTA/IDF1/AssA считаются стандартным TrackEval. Вывод:
- X-метрики (HOTA/IDF1/AssA между камерами) показывают итоговое качество;
- pc-метрики (среднее по камерам) показывают качество внутри камеры; камеры без людей не учитываются (иначе CHIRLA oracle pc HOTA был 0.714 вместо 1.000).
- Рамки считаются в пространстве изображения, а не в BEV, и без маски зоны разметки: сравнивать с литературой (IDF1 на полу 86-96 на WILDTRACK) напрямую нельзя.

### 5.3 Кластерные метрики и handover (уровень треклетов)

| Метрика | Смысл | Подводный камень |
|---|---|---|
| pairwise P/R/F1 | Доля правильных пар треклетов в одном кластере | Считать **по треклетам, а не по рамкам**: иначе один длинный треклет перевешивает десятки коротких (ревью M4; F1 garden2 0.719 → 0.675 после исправления) |
| B-Cubed F, purity | Чистота и полнота кластеров | — |
| ложные слияния / разбиения | Сколько ID слепили разных людей / раздробили одного | Читать вместе с числом ID против GT (22/9, 63/41) |
| **handover** | Доля переходов человека между камерами, где ID сохранился | Самая «бизнесовая» метрика: 0.23 (WILDTRACK all) - 0.98 (garden1 oracle) |
| ReID rank-1/rank-5/mAP | Санити-проверка эмбеддингов по портретам треклетов | Не заменяет ассоциацию: oracle CHIRLA r1 0.69 при IDF1 0.77 |

### 5.4 Oracle-абляции: отделить ошибки ассоциации от ошибок детекции/трекинга

- **Oracle** - треклеты берутся из разметки (детектор и трекер идеальны): видна только ошибка ассоциации. **Full** - весь путь из видео.
- Разрыв oracle → full: terrace -0.50, garden2 -0.40, WILDTRACK test -0.34, AIC24 -0.28, CHIRLA -0.01 (там ограничивает именно межкамерная ассоциация: сидящие люди, похожая одежда).
- Тот же приём применим в других областях: подменяйте выход предыдущей стадии разметкой и смотрите, какая стадия съедает качество (детекция → трекинг → ассоциация → постобработка).
- Оценка «потолка»: идеальная ассоциация тех же треклетов на garden1 даёт X IDF1 0.80, достигнуто 0.64.

### 5.5 Протокол tune/test, абляции, статистика

- Пороги, веса, профиль выбираются **только на tune**, а test используется для итога и абляций «вклад блока» (ничего не настраивает).
- Сплиты: по времени (первые 90 % кадров / последние 10 %; половина/половина) или по последовательностям (garden1 = tune, garden2 = test; seq_001/seq_002 в CHIRLA).
- **Знаки tune и test расходятся** (геометрия AIC24: +0.03 на tune, -0.07 на test). Нужны несколько последовательностей и доверительные интервалы; в проекте каждая цифра - один прогон, разброс соседних конфигов ±0.015 HOTA (T13a).
- Малые выборки шумят: CHIRLA - 9 человек, AIC24 оценивался на 4 из 10 камер.
- Гипотезы фиксируются до прогона (H1-H10), вердикт - с номером прогона #N. Итог по REPORT: полностью подтверждены H1 и H5, частично H2 и H8, не подтверждены H4, H6, H7, отвергнута H3, не проверялись H9 и H10.
- **Ресурсные метрики:** wall time, FPS (кадры всех камер / время холодного прогона), p50/p95 по стадиям, пиковая RSS (0.8-1.3 ГБ), CPU. Время стадии брать из `manifest.json` кеша.
- **Кривая по порогу:** P/R/IDF1 как функция порога решателя (приближённая: постобработка выбрана при одном пороге).

| Тема | RU | EN | Видео |
|---|---|---|---|
| Эксперименты ML: ловушки и воспроизводимость | [Частые ловушки в экспериментах ML (Хабр)](https://habr.com/ru/articles/894220/); [Как эффективно проводить эксперименты (Хабр)](https://habr.com/ru/articles/554852/) | [MLflow (GitHub)](https://github.com/mlflow/mlflow) | ▶RU [MLflow: трекинг экспериментов (Колесников)](https://www.youtube.com/watch?v=CJ16eTcgPvc); ▶RU [MLflow: версионирование экспериментов (OTUS)](https://www.youtube.com/watch?v=oKAFRLH4PUI) |
| Оценка MTMC / AI City | RU: нет | [AIC24 data and evaluation](https://www.aicitychallenge.org/2024-data-and-evaluation/); [8th AI City Challenge](https://arxiv.org/abs/2404.09432) | ▶EN [MTMC demo AIC 2019](https://www.youtube.com/watch?v=WS2Q6yhMxdI) |

---

## 6. Датасеты

| Набор | Тип | Камеры | GT | Лицензия/доступ | Роль в проекте | Ссылки |
|---|---|---|---|---|---|---|
| **EPFL Terrace** | перекрытие, реальный, 360×288 | 4 | треки людей, калибровка | условия оригинального набора EPFL (исследование) | dev (`all`), oracle 1.000 | [EPFL CVLAB](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/) |
| **CAMPUS** (Garden1/2) | частичное перекрытие, 1080p, шумная калибровка | 4 | рамки + ID | MIT (README набора); прямая загрузка без логина | tune/test | [Bitbucket](https://bitbucket.org/merayxu/multiview-object-tracking-dataset) |
| **WILDTRACK** | сильное перекрытие, 1080p, 7 камер, разметка 2 к/с, GT как проекция цилиндра | 7 | 313 людей | текст лицензии на странице не найден - уточнить у EPFL | толпа; full 0.367 | [EPFL](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/); [бенчмарки](https://opencodepapers-b7572d.gitlab.io/benchmarks/multi-object-tracking-on-wildtrack.html) |
| **AIC24 scene_041** | синтетика (NVIDIA PhysicalAI-SmartSpaces), перекрытие | 10 (взято 4) | 28 людей, 3D | cc-by-4.0, без гейта | tune/test по времени | [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces); [AIC](https://www.aicitychallenge.org/) |
| **CHIRLA** | помещения; профиль «без перекрытия» (на деле камеры 1-4 смотрят в одну комнату) | 7 | 9 людей | gated HF (принять условия), CC BY 4.0 | nonoverlap, H3 | [HF](https://huggingface.co/datasets/bdager/CHIRLA); [GitHub](https://github.com/bdager/CHIRLA) |
| **MTMMC** | реальный, многомодальный, без перекрытия | 16 | ID между камерами | подписать соглашение | недоступен без согласования | [сайт](https://sites.google.com/view/mtmmc); [arXiv](https://arxiv.org/abs/2403.20225) |
| **CITIC-MTMC** | помещения, 5 видов, не перекрываются (по README) | 5 | всего 2 человека, GT - позиция на плане | открыт на Zenodo (исследование проекта проверило; curl-у 403) | рассмотрен, мало данных | [GitHub](https://github.com/GTEC-UDC/citic-mtmc-dataset); [Zenodo](https://zenodo.org/records/17512043) |
| **MEVID** | многоракурсный, реальный | — | да | см. страницу набора | рассмотрен | [GitHub](https://github.com/Kitware/mevid) |
| **DukeMTMC** | - | 8 | - | **снят автором** | не использовать | [Exposing.ai](https://exposing.ai/duke_mtmc/) |
| NLPR_MCT | без перекрытия | — | да | сайт офлайн (архив) | недоступен | [Wayback](https://web.archive.org/web/2022/http://mct.idealtest.org/Datasets.html) |
| ReID-кропы: Market-1501, MSMT17, CUHK03 | кропы людей | — | ID | исследовательские | обучение ReID | [MSMT17 (PTGAN)](https://arxiv.org/abs/1711.08565); [Хабр: датасеты ReID](https://habr.com/ru/companies/recognitor/articles/505694/) |

- **Синтетика** (AIC24/PhysicalAI): быстрая замена, но знаки эффекта на ней могут не совпасть с реальными данными (геометрия: tune +0.03, test -0.07).
- **Практика загрузки:** большие наборы медленно качаются через медленный канал - серверу лучше скачивать самому; формат кадров/нумерация (0-based против 1-based) - первая причина багов (`frame_id` в проекте 0-based, общее время `ts = (frame_id + frame_offset)/fps`).
- Свои данные: «Данные → Загрузить свои видео» (N видео + калибровка JSON + GT CSV `camera_id, frame_id, person_id, x, y, w, h`).
- Исследование по датасетам: `agent-dump/inputs/research/datasets.md`, `nonoverlap_datasets.md` (в проекте лежат локально).

---

## 7. Инженерные практики ML-платформы и CPU-инференс

### 7.1 Практики платформы (что, когда, как у нас, подводные камни)

| Практика | Что это | Когда нужна | Как в проекте | Подводные камни | RU | EN |
|---|---|---|---|---|---|---|
| **Контракты данных на pandera** | Схемы таблиц (`detections`, `tracklets`, `global_tracks`...) проверяются при записи и чтении | Несколько авторов/агентов пишут стадии независимо; формат - главный риск | `io/schemas.py`, только `write_table/read_table`; граница «две разработчика - контракт треклетов» | Схема в одном месте, иначе расходится; проверять только на границах стадий | [Проверка Pandas через Pandera (Хабр)](https://habr.com/ru/articles/658473/) | [pandera docs](https://pandera.readthedocs.io/en/stable/) |
| **Реестр компонентов** | `@register(category, name)`, у класса `Params(BaseModel)` и `LICENCE` | Нужно заменять детектор/ReID/решатель конфигом | `registry.py`; модули-плагины перечислены в `PLUGIN_MODULES`, имена файлов совпадают | Мёртвые плагины (ревью: `ultralytics_yolo` без модуля); лицензия компонента видна в каталоге | [Pydantic: валидация (▶Тинькофф)](https://www.youtube.com/watch?v=_5kCjRj6bng) | [Pydantic](https://docs.pydantic.dev/latest/) |
| **Конфиги слиянием** | `base.yaml` + `profiles/` (overlap, nonoverlap) + `datasets/` + overrides из UI | Профили сценариев на общем ядре | `load_config(dataset, profile, overrides)` → `RunConfig`; sweep - ключ конфига + список значений | Вес 0 должен отключать стоимость целиком вместе с cannot-link | RU: нет | [uv docs (окружение и зависимости)](https://docs.astral.sh/uv/) |
| **Контент-адресный кеш стадий** | Каталог `cache/v1/<ds>/<seq>/<stage>-<hash8>/` с `manifest.json` и отпечатком входов | Эксперименты с дешёвой ассоциацией поверх дорогих стадий | Ключ = хеш параметров стадии + входов; повтор ассоциации и оценки 2-10 с вместо 12-19 мин | Ключ должен включать содержимое последовательности и `foot_point` (ревью M2); общий tmp-каталог у параллельных процессов (L9) | [DVC (Хабр)](https://habr.com/ru/articles/761072/); [ML-пайплайн как конечный автомат (Хабр)](https://habr.com/ru/articles/1057548/) | [DVC docs](https://dvc.org/doc) |
| **Очередь задач + SSE** | Один воркер берёт задачи из SQLite, прогресс идёт по Server-Sent Events | Тяжёлые прогоны на малой машине; прогресс в UI | FastAPI (api) + один воркер; SQLModel; `GET /api/jobs/.../events` | Гонка отмены API ↔ воркер (L1); при отмене `running` не должен превратиться в `cancelled` задним числом; не запускать тяжёлое параллельно (RAM-guard) | [SSE: пример (Хабр)](https://habr.com/ru/articles/519982/); [WebSocket, SSE, Long Polling (Хабр)](https://habr.com/ru/articles/812693/) | [MDN: Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events); [SQLite WAL](https://www.sqlite.org/wal.html) |
| **SQLite WAL** | Режим журнала, читатели не блокируют запись | Лёгкая БД состояния, один писатель | `db/`, SQLModel | Файл `-wal` растёт без читателей; для API и воркера - разные процессы | [Бэкап SQLite и WAL (Хабр)](https://habr.com/ru/articles/1094894/) | [SQLModel](https://sqlmodel.tiangolo.com/) |
| **Синхронный видеоплеер** | Сетка камер на одном таймлайне с наложением рамок и глобальных ID, режимы «предсказание / GT / расхождения» | Отладка ассоциации глазами | Окно 8 с, лимит 300 кадров; прокси-видео в H.264 для браузера | При fps > 37 хвост окна молча обрезается (L11); Chromium в Playwright без H.264 | RU: нет | [Playwright docs](https://playwright.dev/docs/intro) |
| **Дашборд экспериментов** | Таблица прогонов, сравнение с дельтами, «Ошибки» (false merge/split с миниатюрами), кривая порога | Перебор гипотез, защита результатов | Страницы «Эксперименты», «Прогон», «Ошибки», «Плеер», «Ресурсы» | `/errors` медленный на больших GT (L4), кеш GT в API съедает RAM (L5) | [Введение в MLflow (Хабр)](https://habr.com/ru/articles/828458/) | [MLflow](https://github.com/mlflow/mlflow) |
| **Генерация типов фронта из OpenAPI** | `api/schemas.py` → `openapi.json` → TypeScript | Два разработчика на API и UI | `web/openapi.json` → типы | Комментарии UI и код расходятся (UI обещал MOT txt, а API принимал CSV) | [FastAPI (рус. документация)](https://fastapi.tiangolo.com/ru/) | [Vite](https://vite.dev/guide/) |
| **Тесты на реальных данных по маркерам** | `@pytest.mark.data/slow/e2e`; `make check` = ruff + mypy + pytest без маркеров | CI на слабой машине | 198 unit-тестов, e2e на Playwright | Тест не должен падать из-за чужой незакоммиченной работы; тестовые функции в проде (удалить) | [Тестирование ML-систем (Хабр)](https://habr.com/ru/articles/680822/); [Playwright (Хабр)](https://habr.com/ru/articles/652919/) | [Playwright](https://playwright.dev/docs/intro) |

- Видео: ▶RU [SSE в FastAPI (OTUS)](https://www.youtube.com/watch?v=XXEiEmefK5M); ▶RU [FastAPI: полный курс](https://www.youtube.com/watch?v=8Zrt6VOBI_Y); ▶EN [Streaming in React with SSE and FastAPI](https://www.youtube.com/watch?v=hOAAg1WaZh8); ▶EN [Pandera, SciPy 2020](https://www.youtube.com/watch?v=PxTLD-ueNd4); ▶RU [DVC: управление данными и моделями (OTUS)](https://www.youtube.com/watch?v=9ff4xU9zHGw).

### 7.2 CPU-инференс

| Решение | Суть | Результат в проекте | RU | EN |
|---|---|---|---|---|
| **OpenVINO (нативный)** | Рантайм Intel, горячий путь (детектор, ReID) | **В 3.5-7 раз быстрее ONNX Runtime для OSNet** на i5-12450H; режимы LATENCY/THROUGHPUT; кеш компиляции `CACHE_DIR` | [ONNX Runtime, OpenVINO и TVM (Selectel, Хабр)](https://habr.com/ru/companies/selectel/articles/696782/); [Ускорение инференса с OpenVINO (Домклик, Хабр)](https://habr.com/ru/companies/domclick/articles/948398/) | [OpenVINO docs](https://docs.openvino.ai/); [GitHub](https://github.com/openvinotoolkit/openvino); [notebooks](https://github.com/openvinotoolkit/openvino_notebooks) |
| **ONNX Runtime** | Универсальный интерчейндж; запасной вариант | На 15-20 % быстрее для YOLOX на этой машине (по оценке REPORT §10) | то же | [ORT docs](https://onnxruntime.ai/docs/); [GitHub](https://github.com/microsoft/onnxruntime) |
| **INT8 (NNCF)** | Пост-тренировочная квантизация | FP16 на 12-м поколении Intel выигрыша не даёт; INT8 - рычаг; не включали в PoC | [INT8 и Throughput mode, Intel (Хабр)](https://habr.com/ru/companies/intel/articles/549636/); [YOLOv8 + OpenVINO + INT8 (Хабр)](https://habr.com/ru/articles/822917/) | [NNCF](https://github.com/openvinotoolkit/nncf); [Model Optimization](https://docs.openvino.ai/2025/openvino-workflow/model-optimization.html) |
| iGPU | Задействовать графику Intel | Требует `compute-runtime` (OpenCL) на хосте; на UHD мало выигрыша | — | [OpenVINO docs](https://docs.openvino.ai/) |
| Батчинг кропов, `reid_interval` | Собирать кропы всех камер в один батч; ReID каждые K кадров | ReID - 70-80 % времени кадра в ByteTrack + OSNet (Хабр 2026) | [ByteTrack на GPU, батчинг OSNet (Хабр)](https://habr.com/ru/articles/1059236/) | [Roboflow trackers](https://github.com/roboflow/trackers) |
| Экспорт в ONNX из PyTorch только один раз | PyTorch - в отдельной группе зависимостей `export` | В рантайме нет torch (RAM ≤ 1.3 ГБ) | — | [uv](https://docs.astral.sh/uv/) |

- **Бюджет CPU:** на 4 камерах × 10 к/с детектор YOLO-класса ~30-100 мс/кадр - офлайн норма; реальное время 2-4 камер на ноутбуке - предел. Время холодного прогона: 12-19 мин на 100-200 с видео 4 камер.
- Видео: ▶RU [Ускорение исполнения нейросетей с OpenVINO (ODS, Темнов)](https://www.youtube.com/watch?v=MMhdwDHqVCo); ▶RU [Intel OpenVINO: ускоритель нейронных сетей](https://www.youtube.com/watch?v=IFWdJN2B4gc); ▶RU [Ускоряем нейросети на GPU и CPU: дистилляция и квантование](https://www.youtube.com/watch?v=-2fN7aJDYkw); ▶RU [Быстрый инференс на слабом железе (Яндекс)](https://www.youtube.com/watch?v=LC00xqp2sms); ▶EN [OpenVINO Execution Provider For ONNX Runtime (OpenCV Live)](https://www.youtube.com/watch?v=D9aR7LbHMko).

### 7.3 Фронтенд и инструменты разработки

- React 19 + Vite + TypeScript, типы из OpenAPI; Node 22 нужен для Vite/Tailwind 4 (системный Node 18 не годится). Playwright e2e с Chromium без H.264 - видео в e2e проверять через прокси-кодек/Chrome.
- Python: `uv` (воспроизводимые окружения, группы зависимостей), `ruff`, `mypy`; `make check` как единая точка проверки.
- Видео: ▶RU [Pydantic (Шумейко)](https://www.youtube.com/watch?v=zTSRygNQ_Fw); ▶EN [Docker non-root UID/GID (Nick Janetakis)](https://www.youtube.com/watch?v=sXfaogNlc7Y).

---

## 8. Развёртывание и эксплуатация

| Тема | Что делали / правило | Подводный камень | RU | EN |
|---|---|---|---|---|
| **Docker, non-root** | Один образ для `api` и `worker`, многоступенчатая сборка (Node 22 → фронт, `python:3.12-slim` + `uv`), запуск не от root | У non-root пользователя нет домашнего каталога с правом записи: HF-кеш, `~/.cache` падают, нужен `HOME` с записью | [Root в контейнере (Хабр)](https://habr.com/ru/articles/1040300/); [Безопасная сборка Docker-образов (Хабр)](https://habr.com/ru/articles/813047/) | [Docker best practices](https://docs.docker.com/build/building/best-practices/); [Rootless](https://docs.docker.com/engine/security/rootless/) |
| **Тома с хоста** | `data/ cache/ models/ outputs/ db/` монтируются, не копируются в образ | Права на тома (UID хоста ≠ UID контейнера); веса качаются на хосте (`make setup`), torch не в образе | [Docker для начинающих (Хабр)](https://habr.com/ru/articles/888540/) | [Docker Compose](https://docs.docker.com/compose/) |
| **Лимиты памяти** | `mem_limit`, RAM-guard в пайплайне, ≤ 60-70 % RAM машины; тяжёлые стадии не параллелить | OOM убивает воркер (код 137): проверять `docker stats`, пик RSS в `runtime.json` | — | [Compose reference](https://docs.docker.com/compose/) |
| **nginx + HTTPS** | Обратный прокси, TLS через `certbot certonly --nginx`, basic auth на нестандартном порту | nginx 1.24 не знает `http2 on` (писать `listen 443 ssl http2`); сертификат продлевать по таймеру | RU: [nginx: auth_basic](https://nginx.org/ru/docs/http/ngx_http_auth_basic_module.html) | [Certbot](https://certbot.eff.org/); [nginx auth_basic](https://nginx.org/en/docs/http/ngx_http_auth_basic_module.html) |
| **Медленный канал** | Не загружать датасеты через браузер: сервер сам скачивает (кнопка «Скачать» ставит задачу) | Starlette сохраняет multipart целиком во временный файл до проверки лимитов (L3) | — | — |
| **Публикация в GitHub без секретов** | Перед открытием репозитория вычистить из истории секреты, домены, пароли basic auth; `.gitignore` для `data/`, `cache/`, `models/`, `outputs/`, `db/` | Правило `data/` в `.gitignore` скрыло и `src/mtmct/data` - якорить `/data/`; чистка истории (`git filter-repo`) раньше публикации | [Что знать о лицензиях и Copyright (Хабр)](https://habr.com/ru/articles/1044986/) | [HF: токены доступа](https://huggingface.co/docs/hub/security-tokens) |
| **Доступ только с localhost** | `127.0.0.1:8080` в compose; наружу - только через nginx с auth | Внешний доступ без auth открывает загрузку файлов и запуск прогонов | — | — |

- Видео: ▶RU [nginx + SSL: certbot, Let's Encrypt](https://www.youtube.com/watch?v=hzMVWTSfuwg); ▶RU [Docker: полный курс для начинающих](https://www.youtube.com/watch?v=_uZQtRyF6Eg); ▶EN [Docker volumes: permission denied](https://www.youtube.com/watch?v=iODjCjHSpVU); ▶EN [Docker non-root UID/GID](https://www.youtube.com/watch?v=sXfaogNlc7Y).

---

## 9. Агентная разработка (мастер / исполнители / ревьюер)

### 9.1 Схема

```
владелец (решения, ограничения)
   │
мастер (держит контекст и решения; HANDOFF.md, PLAN.md, TASKS.md)
   ├─ исполнитель T1 ... Tn (Claude Opus/Sonnet; непересекающиеся зоны файлов)
   └─ ревьюер (после итерации: режет переусложнение, ищет баги, read-only)
```

- Все материалы агентов лежат в `agent-dump/` (gitignored); бизнес-артефакты в `artefacts/` (в git).
- **HANDOFF.md** - единственный источник правды: цель, решения владельца, архитектурные инварианты, стенд (RAM, Node, ffmpeg), правила исполнителей, журнал решений. **TASKS.md/PLAN.md** - задачи и интерфейсы. Отчёт каждого исполнителя - `reports/Tn_report.md`.
- Зоны файлов: исполнитель не трогает общие файлы (`io/`, `registry.py`, `config.py`, `components.py`, `api/schemas.py`, `configs/base.yaml`, `pyproject.toml`, `Makefile`, `tests/conftest.py`) - описывает правку в отчёте, вносит мастер. Новые зависимости - `uv add` самим исполнителем с перечислением.
- Только Claude-субагенты (по решению владельца, без Codex); коммиты без строк атрибуции; решения делегированы мастеру; «две честные попытки не помогли - остановись и зафиксируй доказательства».

### 9.2 Что сработало

| Практика | Эффект |
|---|---|
| Единый HANDOFF с «архитектурными решениями - не пересматривать без мастера» | Исполнители не спорили об архитектуре; новые люди (агенты) подключались без пересказа |
| Непересекающиеся зоны + общие файлы только у мастера | Нет конфликтов слияния при параллельной работе |
| TDD (тест → красный → код → зелёный) + `verification-before-completion` | 198 unit-тестов, ревью нашло 2 High / 4 Medium / 12 Low, а не поломку на каждом шаге |
| **Независимое ревью read-only после волны** | Нашло баги, которые не ловили тесты: H1 time-cost запрещал 16 из 16 настоящих пар на CHIRLA test; H2 вес 0 не отключал cannot-link; M1 интерполяция на необработанные кадры; M2 неполный ключ кеша |
| Проверки на реальных данных отдельным маркером и малые входы в тестах | Не убивали RAM машины; отладка укладывалась в 5 ГБ свободной памяти |
| Ponytail («самое маленькое полное изменение») + список «удалить/упростить» в ревью | ~1300 строк mock-симулятора и мёртвые ветки убраны; кодовая база не раздулась |
| Отчёты агентов с доказательствами и номерами прогонов | Вердикты по гипотезам воспроизводимы по `#N` |
| Мастер выбирает режим эксперимента: oracle, чтобы развести узкие места | Дешёвые итерации (секунды) |

### 9.3 Что не сработало / риски

| Проблема | Как проявлялась | Что делать |
|---|---|---|
| Исполнитель правил общие файлы или «заплатки под тест» | Откаты; эвристики под датасет вне адаптера | Явное правило в брифе: «нарушения откатываются»; ревьюер проверяет дифф |
| Агенты берут первый ответ как истину | Геометрия выбиралась на oracle (показывала +), на full она не помогала | Выбор конфигурации по режиму, на котором будет жить решение (full tune) |
| Контракт документации/UI расходится с кодом | Подсказка UI «MOT txt», а парсер ждал 7-колоночный CSV | Ревью сверяет подсказки и тексты UI с API |
| Две параллельные тяжёлые задачи | Нехватка RAM, зависание | Правило в HANDOFF: никогда не запускать тяжёлый инференс параллельно |
| Незакоммиченная работа соседнего агента ломает тест | Ложный «красный» в чужой зоне | Ревьюер фиксирует и исключает из счёта; мастер коммитит порционно |
| Перегруженный контекст мастера | Решения теряются | Журнал решений в HANDOFF/RESUME, отчёты в файлах, а не в чате |

### 9.4 Skills и методики

| Что | Для чего | Ссылки |
|---|---|---|
| **superpowers** (obra) | TDD, systematic-debugging, verification-before-completion, requesting-code-review, writing-plans, subagent-driven-development, using-git-worktrees | [GitHub](https://github.com/obra/superpowers); ▶EN [Jesse Vincent с Tim O'Reilly](https://www.youtube.com/watch?v=c5AxRQJy1eA) |
| **mattpocock/skills** | Короткие skills для инженеров: grilling (допрос плана), TDD, диагностика | [GitHub](https://github.com/mattpocock/skills); ▶EN [mattpocock/skills: workflow end-to-end](https://www.youtube.com/watch?v=M6mYodf0dJM); ▶EN [5 skills I use every day](https://www.youtube.com/watch?v=EJyuu6zlQCg) |
| **ponytail** | Самое маленькое полное изменение, список «чего не сделал» | в проекте как skill `ponytail:ponytail` |
| **agent-fleet** | Оркестрация «мастер → исполнители → ревьюер» | в проекте как skill `agent-fleet` |
| Документация Claude Code по субагентам | Описание агентов, делегирование, контекст | RU: [Хабр: Мой фреймворк для агентной разработки с Claude Code](https://habr.com/ru/articles/1022050/); [Хабр: советы по Claude Code](https://habr.com/ru/articles/929624/); ▶RU [Гайд по сабагентам в Claude Code](https://www.youtube.com/watch?v=gbPKKrV-JGo); EN: [Sub-agents docs](https://docs.anthropic.com/en/docs/claude-code/sub-agents); [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents); [Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system); [Context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents); ▶EN [Claude Code Subagents and Multi Agent Workflows](https://www.youtube.com/watch?v=ivwh7B-M1NA) |

---

## 10. Подводные камни и уроки: «симптом → причина → решение»

### 10.1 Качество и методика

| Симптом | Причина | Решение |
|---|---|---|
| Межкамерный IDF1 0.4-0.7 при «хорошем» ReID | Узкое место - внутрикамерные треклеты (DetA 0.5-0.67, ~10 фрагментов на человека) | Oracle-режим для диагностики; чинить детектор/трекер/refine, а не ассоциацию |
| Ложные слияния людей в одной ID | Эмбеддинг треклета усредняет двух людей после обмена ID в толпе | Разрез треклета по скачку внешности (`split_thr 0.75`, окно 7): +0.16 IDF1 |
| Геометрия улучшает oracle, но ухудшает full | Шум точки ног и калибровки на длинных/грязных треклетах | Мягкий вклад (вес 0.25-0.5), медиана расстояния, выбор на full tune; точка ног по позе (H9) - гипотеза |
| Жёсткий gate по гомографии режет настоящие пары | Калибровка CAMPUS: медиана 1-1.7 м, p90 до 9 м | `hard: false`, gate 5 м, вес 0.5 |
| Окна времени «лучше» в теории, хуже на данных | На CHIRLA камеры 1-4 смотрят в одну комнату, одновременное наблюдение даёт Δt ≈ 0 и cannot-link на настоящих парах | Модель связей: одновременно видимые = «нет информации»; визиты как непрерывные сегменты; `overlap=ignore` |
| Абляция «стоимость выключена» не меняет картину | Стоимость с весом 0 всё равно накладывала cannot-link | Вес 0 отключает стоимость целиком, вместе с запретами |
| Пороги «поплыли» после замены ReID-модели | Разные шкалы косинуса моделей | Таблица порогов по модели; подбирать на tune заново |
| Re-ranking увеличил число ID с 18 до 37 | k-reciprocal меняет шкалу расстояний | Отдельный порог для re-ranking |
| Хороший результат на tune, плохой на test | Переобучение на одну последовательность; знак эффекта расходится | Несколько последовательностей, доверительные интервалы, абляции только на test без настройки |
| pairwise F1 завышен | Считали по рамкам: длинный треклет весит как n² пар | Метрики на уровне треклетов |
| Среднее по камерам занижено | Камеры без людей входили в среднее | Исключать камеры без GT и предсказаний |
| Оценка WILDTRACK не сходится с литературой | Разметка проекцией 3D-цилиндра в зоне 12×36 м, IDF1 на плоскости пола в литературе | Сравнивать только с оговоркой; маска зоны разметки в планах |
| Интерполяция даёт рамки «из воздуха» | Дробный шаг кадров 25→10 к/с | Интерполировать только на обработанные кадры сетки |

### 10.2 Инженерия и среда

| Симптом | Причина | Решение |
|---|---|---|
| `cv2` ломается при установке | `trackers` тянет `opencv-python`, а в окружении есть `opencv-python-headless` (или наоборот): конфликт модулей | В проекте один пакет `opencv-python`; headless не ставить |
| TrackEval падает на `np.float` / `np.int` | NumPy 2 удалил алиасы | Вендорная копия `third_party/TrackEval` с патчем |
| AGPL-эффект | `ultralytics`/`boxmot` под AGPL-3.0: сетевой сервис = раскрытие исходников | Базовый путь только permissive (YOLOX, ByteTrack, OSNet); AGPL - опциональные адаптеры |
| Веса insightface нельзя в продукт | Модели insightface - только некоммерческие | Лицо - опциональный признак за флагом, по умолчанию выключен |
| OSNet на ORT медленный | ORT-CPU для сверточных ReID-сетей уступает OpenVINO | OpenVINO в горячем пути (3.5-7× быстрее), ORT запасной |
| iGPU не виден OpenVINO | Нет `compute-runtime` (OpenCL) | Установить пакеты `intel-opencl-icd`/compute-runtime на хост |
| Vite/Tailwind 4 не собираются | Системный Node 18 | Node 22 в `~/.local/opt/node/bin` и в Docker-образе сборки |
| e2e не воспроизводит видео | Playwright Chromium без кодека H.264 | Проверять через Chrome; прокси-видео в H.264/MP4 для UI; видеоплеер вне e2e |
| `src/mtmct/data` пропал из git | `.gitignore: data/` совпал с кодом | Якорить: `/data/` |
| `ffmpeg` не найден | В системе его нет | `imageio_ffmpeg.get_ffmpeg_exe()` |
| Нехватка RAM | Параллельные тяжёлые процессы; кеш GT в API (`lru_cache`) | Один воркер, RAM-guard, `maxsize=2` для GT, малые тестовые входы |
| Повторная загрузка custom-набора навсегда блокируется | Копии видео остаются в `data/custom/<name>` после сбоя подготовки | В `except` удалять и `data/custom/<name>`; проверять GT сразу (422) |
| Старые результаты после смены `cameras.json` | Ключ кеша не включал содержимое последовательности и `foot_point` | Включить отпечаток входов в ключ (`fingerprint` в manifest) |
| Контейнер падает при записи кеша HF | Non-root без домашнего каталога | Задать `HOME`/`HF_HOME` на запись |
| `http2 on` отвергается nginx | Версия nginx 1.24 | `listen 443 ssl http2;` |
| Медленная загрузка больших видео | Канал | Сервер скачивает датасет сам (задача «Скачать») |
| Утечка секретов при публикации репозитория | Домены/пароли в истории | Почистить историю до публикации, secrets вне репозитория |
| Sweep в UI отображается как `failed` | `DetachedInstanceError` в SQLAlchemy | Закрепить `done` на воркере (исправлено) |

---

## 11. Правовые и этические аспекты (не юридическая консультация)

### 11.1 Что такое «биометрия» в нашем случае

| Тема | Вывод проекта | Источники |
|---|---|---|
| **Сопоставление лиц** | Биометрические персональные данные: специальная категория во всех режимах (152-ФЗ, GDPR ст. 9, AI Act). В проекте - только опциональный признак за флагом, по умолчанию выкл. | RU: [152-ФЗ (КонсультантПлюс)](https://www.consultant.ru/document/cons_doc_LAW_61801/); [Что нужно знать юристу про биометрические ПДн (law.ru)](https://www.law.ru/article/28014-chto-nujno-znat-yuristu-kompanii-pro-biometricheskie-personalnye-dannye); [Сравнение GDPR, 152-ФЗ, PIPL (Хабр)](https://habr.com/ru/articles/1023326/) / EN: [GDPR ст. 9](https://gdpr-info.eu/art-9-gdpr/); [GDPR ст. 9 на русском](https://gdpr-text.com/ru/read/article-9/) |
| **ReID по телу без идентификации** | Ниже риск: идентификатор не привязывается к личности, эмбеддинги нельзя считать «биометрией» автоматически, но при привязке к ФИО или при уникальности вектора для конкретного лица оценка меняется. Нужна оценка конкретного кейса | RU: [Реидентификация 2.0: без лиц, на краю (Хабр)](https://habr.com/ru/companies/kryptonite/articles/704914/) / EN: [EDRi: удалённая биометрическая идентификация](https://edri.org/our-work/remote-biometric-identification-a-technical-legal-guide/) |
| **EU AI Act** | Удалённая биометрическая идентификация в реальном времени в публичных местах - запрещённая практика (ст. 5, с исключениями); системы повышенного риска - требования к данным и прозрачности | RU: [EU AI Act и санкции для российских IT (Хабр)](https://habr.com/ru/articles/1015812/) / EN: [Статья 5](https://artificialintelligenceact.eu/article/5/); [Регламент 2024/1689 (EUR-Lex)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) |
| **Хранение эмбеддингов** | Не хранить эмбеддинги реальных людей без согласия; ограничение срока; обезличивание; доступ с аутентификацией | RU: [Роскомнадзор: ПДн (Хабр)](https://habr.com/ru/articles/946790/) |
| Видео: | ▶RU [152-ФЗ: полный разбор](https://www.youtube.com/watch?v=MrqMhlvKUJ4); ▶RU [Биометрия сотрудников и штрафы по 152-ФЗ](https://www.youtube.com/watch?v=MFixCL-wp7Y); ▶RU [152-ФЗ с сентября 2025 (VK Tech)](https://www.youtube.com/watch?v=Xo1w9oJ9KOg); ▶EN [GDPR Article 9 Explained](https://www.youtube.com/watch?v=UrR-cQB4CyM); ▶EN [EU AI Act: Biometric Verification vs Identification](https://www.youtube.com/watch?v=e06OB3nCMpA) | |

### 11.2 Лицензии моделей, кода и данных

| Категория | Риск | Правило |
|---|---|---|
| AGPL (ultralytics, boxmot, YOLO v8/10/11/12/26) | Раскрытие исходников для сетевых сервисов; веса несут те же условия | Опциональные адаптеры, в базовом пути не использовать; для продукта - YOLOX/RT-DETR/D-FINE (Apache) |
| GPL (YOLOv6/7/9, `leidenalg`, `igraph`) | Распространение = раскрытие | Избегать в поставляемом продукте |
| NC-лицензии (insightface-модели, CR-FIQA) | Коммерческое использование запрещено | Не поставлять |
| «Веса MIT, данные исследовательские» (OSNet на Market/MSMT17/Duke) | Происхождение обучающих данных | Для коммерческого развёртывания - провести due diligence или дообучить на своих данных |
| Датасеты | DukeMTMC снят, MTMMC и CHIRLA - по соглашению, MMPTRACK - NC | Проверять условия до загрузки; не публиковать данные в репозитории |
| Публикация проекта | В репозиторий не попадают данные/кеш/веса/секреты | `.gitignore` с якорями, чистка истории |

- Этика данных: видео с людьми - только датасеты с согласием/открытой лицензией; в UI скрывать лица при показе третьим лицам (в плане нет, рекомендация); журналировать доступ; не использовать для оценки поведения.
- Ссылки: [Лицензии моделей детекции](https://robotinaction.tech/model-licenses.html); [Выбор детектора и лицензия (AI Рассвет)](https://airassvet.ru/articles/yolo-rt-detr-rf-detr-d-fine-sravnenie); [insightface model zoo (GitHub)](https://github.com/deepinsight/insightface).

---

## 12. Что изучать дальше

| Направление | Зачем в этом проекте | Первые шаги | RU | EN |
|---|---|---|---|---|
| **Склейка фрагментов внутри камеры** (до межкамерной) | Главное узкое место: ~10 треклетов на человека | Кластеризация по внешности внутри камеры, вес портрета по длине треклета, дополнительный сигнал разреза (скачок положения/размера) | RU: нет | [CAMELTrack](https://arxiv.org/abs/2505.01257); [ReidTrack](https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Specker_ReidTrack_Reid-Only_Multi-Target_Multi-Camera_Tracking_CVPRW_2023_paper.pdf) |
| **GNN-ассоциация** | Обучаемые связи вместо суммы стоимостей | MPNTrack → UMPN/ReST на тех же треклетах (смена решателя в реестре) | RU: нет | [MPNTrack](https://arxiv.org/abs/1912.07515); [UMPN](https://arxiv.org/abs/2507.08494); [ReST](https://arxiv.org/abs/2308.13229) |
| **Сквозные MCTR / MOTIP** | Online-версия без эвристик | Прочитать MOTIP/MCTR, воспроизвести на GPU-сервере (CPU непригоден) | RU: нет | [MOTIP](https://arxiv.org/abs/2403.16848); [MCTR](https://openaccess.thecvf.com/content/WACV2025W/RWS/papers/Niculescu-Mizil_MCTR_Multi_Camera_Tracking_Transformer_WACVW_2025_paper.pdf); [MCTR (GitHub)](https://github.com/necla-ml/mctr); [MOTRv2](https://arxiv.org/abs/2211.09791) |
| **BEV / раннее слияние** | Максимум точности при точной калибровке (WILDTRACK IDF1 ~95) | Прочитать MVDet → EarlyBird → TrackTacular → ATLASFusion; сравнить на WILDTRACK | RU: [Яндекс: BEV и гомография (Хабр)](https://habr.com/ru/companies/yandex/articles/844126/) | [MVDet](https://arxiv.org/abs/2007.07247); [TrackTacular](https://arxiv.org/abs/2403.12573); [ATLASFusion](https://arxiv.org/abs/2509.08421); [MCBLT](https://arxiv.org/abs/2412.00692) |
| **Дообучение под домен** | Порог ReID привязан к модели; домен-gap | Самообучение OSNet x0_5 на псевдо-ID треклетах; OMZ-0287 как модель по умолчанию после подбора порогов | RU: нет | [FastReID](https://github.com/JDAI-CV/fast-reid); [Bag of Tricks](https://arxiv.org/abs/1903.07071) |
| **Точка ног по позе (H9)** | Убрать шум геометрии | RTMPose/ViTPose, сравнение с низом рамки | RU: нет | [RTMPose](https://arxiv.org/abs/2303.07399); [mmpose](https://github.com/open-mmlab/mmpose) |
| **Детектор: прямоугольный вход, маска зоны, ORT** | DetA 0.28 на WILDTRACK | Вход 1088×608 без паддинга, маска GT-зоны | [YOLOv11 и RF-DETR в видео (Хабр)](https://habr.com/ru/articles/1018062/) | [D-FINE](https://arxiv.org/abs/2410.13842); [RT-DETR](https://arxiv.org/abs/2304.08069) |
| **Онлайн-режим (H10)** | Реальное время | Скользящее окно, инкрементальная кластеризация | [Хабр: ByteTrack на GPU и батчинг](https://habr.com/ru/articles/1059236/) | [OCMCTrack](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Specker_OCMCTrack_Online_Multi-Target_Multi-Camera_Tracking_with_Corrective_Matching_Cascade_CVPRW_2024_paper.pdf) |
| **Статистика оценки** | Знаки tune/test расходятся | Бутстреп по последовательностям/окнам, несколько сидов, доверительные интервалы | [Частые ловушки в экспериментах ML](https://habr.com/ru/articles/894220/) | [HOTA](https://arxiv.org/abs/2009.07736) |
| **Лицо + тело (опционально)** | Усиление ReID с юридическим барьером | Только с согласием и юридической оценкой; оценка качества лица | RU: нет | [QME](https://arxiv.org/abs/2508.00053); [Person Recognition at Altitude and Range](https://arxiv.org/abs/2505.04616) |
| **Без перекрытия** | Опыт проекта: окна времени не помогли на CHIRLA | Развитие направления отменено владельцем; опыт зафиксирован в REPORT (H3) и в исследовании датасетов | см. §3.2, §6 | [MICRO-TRACK](https://arxiv.org/abs/2409.03879); [MTMMC](https://arxiv.org/abs/2403.20225) |
