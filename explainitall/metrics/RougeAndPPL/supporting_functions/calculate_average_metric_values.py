def calculate_average_metric_values(calculated_metrics):
    res = {}

    for metric in calculated_metrics:
        sub_metric_average_values = {}
        for text_evaluation_result in calculated_metrics[metric]:
            for sub_metric in text_evaluation_result:
                if sub_metric not in sub_metric_average_values:
                    sub_metric_average_values[sub_metric] = 0.0
                sub_metric_average_values[sub_metric] += text_evaluation_result[sub_metric]
        for sub in sub_metric_average_values:
            sub_metric_average_values[sub] = sub_metric_average_values[sub] / len(calculated_metrics[metric])
        if 'f1' in sub_metric_average_values:
            res[metric] = sub_metric_average_values['f1']
        elif 'value' in sub_metric_average_values:
            res[metric] = sub_metric_average_values['value']
        else:
            res[metric] = 0.0

    return res
