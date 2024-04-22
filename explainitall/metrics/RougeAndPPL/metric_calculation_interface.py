import gradio as gr
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModel, AutoModelForCausalLM

from explainitall.metrics.RougeAndPPL.Metrics import Metric_ppl, MetricRougeL, MetricRougeN
from explainitall.metrics.RougeAndPPL.Metrics_calculator import Metrics_calculator
from explainitall.metrics.RougeAndPPL.create_database import data_table, ENGINE
from explainitall.metrics.RougeAndPPL.helpers import (
    get_max_dataset_version, generate_candidates, calculate_average_metric_values,
    insert_new_record, get_records_from_database, make_dataframe_from_history_records)


class MetricCalculationInterface:
    demo_ = None

    model_checkbox_ = None
    metrics_checkbox_ = None

    calculator_ = None

    model_name_ = None
    model_ = None
    tokenizer_ = None
    model_successfully_loaded_ = None

    dataset_name_ = None
    dataset_version_ = None
    contexts_ = None
    references_ = None
    dataset_successfully_loaded_ = None

    conn_ = None

    def __init__(self):
        self.conn_ = ENGINE.connect()

        self.demo_ = gr.Blocks()

        with self.demo_:
            with gr.Tabs():
                with gr.TabItem("Load"):
                    gr.Markdown("""
                        Загрузите CSV файл с набором данных, который будет использоваться для анализа и оценки качества модели генерации текста</br>
                         Набор данных должен содержать как минимум две колонки:</br>
                        - `context`: текстовый контекст или вводные данные, на основе которых модель будет генерировать текст</br>
                        - `reference`: эталонный текст или ожидаемый ответ модели на заданный контекст </br>
                        Данный набор данных позволит оценить, насколько хорошо модель способна генерировать текст, соответствующий заданному контексту и эталонным ответам
                        Пример файла example_data/metrix.csv
                        """)
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        dataset_file = gr.File(label='Dataset (CSV)')
                                        dataset_title = gr.Text(label='Dataset title',
                                                                info="Это поле обязательно к заполнению. Без указания названия датасета процесс не будет запущен.",
                                                                placeholder="Введите название датасета")

                                    dataset_visualization = gr.Dataframe(label='Dataset',
                                                                         headers=['context', 'reference'],
                                                                         wrap=False,
                                                                         height=500)
                                with gr.Row():
                                    dataset_load_button = gr.Button("Load dataset")
                                    dataset_checkbox = gr.Checkbox(label='dataset loaded', interactive=False)

                        with gr.Row():
                            with gr.Column():
                                model_name_or_path = gr.Text(label='Model name or path',
                                                             placeholder="distilgpt2",
                                                             info="Введите название предварительно обученной модели (например, distilgpt2, gpt2 или sberbank-ai/rugpt3small_based_on_gpt2) или путь к вашей модели")

                                with gr.Row():
                                    model_load_button = gr.Button("Load model")
                                    model_checkbox = gr.Checkbox(label='model loaded', interactive=False)
                        with gr.Row():
                            launch_button = gr.Button("Launch")
                            metrics_checkbox = gr.Checkbox(label='metrics calculated, откройте вкладку Result и обновите данные', interactive=False)

                with gr.TabItem("Result"):
                    with gr.Row():
                        filter_field_dropdown = gr.Dropdown(["None", "model_name", "dataset_name"],
                                                            label='Filter by field')
                        filter_value_text = gr.Text(label='Filter value')
                    refresh_button = gr.Button("Refresh")
                    history_table = gr.Dataframe(
                        headers=['model_name', 'date', 'dataset_name', 'dataset_version', 'PPL', 'R3', 'R5', 'R-L'],
                        label='All Data')

            dataset_load_button.click(self.load_dataset_,
                                      inputs=[dataset_file, dataset_title],
                                      outputs=[dataset_visualization, dataset_checkbox])
            model_load_button.click(self.load_model_,
                                    inputs=[model_name_or_path],
                                    outputs=[model_checkbox])
            launch_button.click(self.calculate_metrics_,
                                inputs=None,
                                outputs=[metrics_checkbox])
            refresh_button.click(self.refresh_history_,
                                 inputs=[filter_field_dropdown, filter_value_text],
                                 outputs=[history_table])

            self.model_checkbox_ = model_checkbox
            self.dataset_checkbox_ = dataset_checkbox
            self.metrics_checkbox_ = metrics_checkbox

    def launch(self):
        self.demo_.launch(share=True, debug=False, server_name="127.0.0.1", inbrowser=True)

    def load_model_(self, model_name_or_path):
        if not model_name_or_path:
            print("Model name or path is required.", model_name_or_path)

        self.model_successfully_loaded_ = False

        self.tokenizer_ = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer_.pad_token is None:
            self.tokenizer_.pad_token = self.tokenizer_.eos_token
        self.model_ = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        self.model_name_ = str(model_name_or_path)

        self.calculator_ = Metrics_calculator(self.tokenizer_)

        self.calculator_.add_metric('PPL', Metric_ppl(self.model_, stride=512))

        self.calculator_.add_metric('R3', MetricRougeN(3))
        self.calculator_.add_metric('R5', MetricRougeN(5))

        self.calculator_.add_metric('R-L', MetricRougeL(3))

        self.model_successfully_loaded_ = True
        return True

    def load_dataset_(self, csvfile, title):
        self.dataset_successfully_loaded_ = False

        print("csvfile", csvfile)
        print("title", title)

        if csvfile is None or title == '':
            return pd.DataFrame(), False
        dataframe = pd.read_csv(csvfile.name, delimiter=',', encoding='utf-8')

        self.dataset_name_ = title
        self.dataset_version_ = get_max_dataset_version(self.dataset_name_, self.conn_, data_table) + 1

        self.contexts_ = list(dataframe['context'].values)
        self.references_ = list(dataframe['reference'].values)

        self.dataset_successfully_loaded_ = True
        return dataframe.head(10), True

    def calculate_metrics_(self):
        if self.dataset_successfully_loaded_ is False or self.model_successfully_loaded_ is False:
            return False

        model = self.model_
        candidates = generate_candidates(model,
                                         self.tokenizer_,
                                         self.contexts_,
                                         model.config.n_positions,
                                         max_new_tokens=10)

        res = self.calculator_.calculate(self.contexts_, self.references_, candidates)
        metric_values = calculate_average_metric_values(res)

        insert_new_record(data_table=data_table,
                          conn=self.conn_,
                          model_name=self.model_name_,
                          dataset_name=self.dataset_name_,
                          dataset_version=self.dataset_version_,
                          metric_values=metric_values)


        return True

    def refresh_history_(self, filter_field, filter_value):
        specific_column_value = None
        if filter_field != [] and filter_field is not None and filter_field != 'None' and filter_field != '':
            specific_column_value = {filter_field: filter_value}

        res = get_records_from_database(data_table, self.conn_, specific_column_value)
        df = make_dataframe_from_history_records(res)

        return df
