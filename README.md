# **ExplainitAll**
**ExplainitAll** — это библиотека для интерпретируемого ИИ, предназначенная для интерпретации генеративных моделей (GPT-like), и векторизаторов, например, Sbert. Библиотека предоставляет пользователям инструменты для анализа и понимания работы этих сложных моделей. Кроме того, содержит модули RAG QA, fast_tuning и пользовательский интерфейс.

---

* [Примеры использования](https://github.com/Bots-Avatar/ExplainitAll/tree/main/examples)
* [Исходный код библиотеки](https://github.com/Bots-Avatar/ExplainitAll/tree/main/explainitall)
* [Документация](https://github.com/Bots-Avatar/ExplainitAll/wiki)

## Модели:

* Дистиллированный [Sbert](https://huggingface.co/FractalGPT/SbertDistil)
* Дистиллированный [Sbert](https://huggingface.co/FractalGPT/SbertSVDDistil) с применением SVD разложения, для ускорения инференса и обучения
* [FRED T5](https://huggingface.co/FractalGPT/FRED-T5-Interp), обученный под задачу RAG, для ответов на вопросы по интепретации генеративной gpt-подобной сети.


---
## Перечень направлений прикладного использования:

Результаты могут применяться в следующих областях: любые вопрос-ответные системы или классификаторы критических отраслей (медицина, строительство, космос, право и т.п.). Типовой сценарий применения, например для медицины следующий: разработчик конечного продукта, такого как например система поиска противопоказаний у лекарств в тесном взаимодействии в заказчиком(врачом, поликлиникой и т.п.) создает набор кластеров тематической области, дообучает трансформерную модель (GPT-like: например, семейств ruGPT3 и GPT2) на текстах вопрос-ответ, и на затем в режиме эксплуатации данной, уже готовой Вопросно-ответной системы подключает библиотеку ExplainitAll для того, чтобы она давал аналитическую оценку – насколько «надежными» и доверенными являются ответы вопросно-ответной системы на основе результата интерпретации – действительно ли при ответе на вопросы пользователя система обращала внимание на важные для отрасли кластеры.

Разработанная библиотека может быть адаптирована как модуль конечного продукта - ассистента врача, инженера-конструктора, юриста, бухгалтера. Для государственного сектора библиотека может быть полезна т.к. помогает доверять RAG системам при ответах по налогам, регламентам проведения закупочных процедур, руководствам пользователей информационных систем, нормативно-правовым актам регулирования. Для промышленных предприятий библиотека применима в работе с регламентами, руководствами по эксплуатации и обслуживанию сложного технического оборудования, т.к. позволяет оценивать учет в ответах QA систем понимание специальных, важных для отрасли сокращений, наименования, аббревиатур, номенклатурных обозначений.


## Характеристики:
* Операционная система Ubuntu 22.04.3 LTS
* Драйвер: NVIDIA версия 535.104.05
* CUDA версия 12.2
* Python 3.10.12


* Процессор AMD Ryzen 3 3200G OEM (частота: 3600 МГц, количество ядер: 4)
* Оперативная память 16 GB 

* Графический процессор
  * Модель: nVidia TU104GL [Tesla T4]
  * Видеопамять 16 GB


* Лицензия [Apache-2.0 license](https://github.com/Bots-Avatar/ExplainitAll/tree/main#Apache-2.0-1-ov-file)
