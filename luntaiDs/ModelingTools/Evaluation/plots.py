import os
from datetime import timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
from itertools import cycle
from bokeh.plotting import figure, show, Figure
from bokeh.models import ColumnDataSource, Dropdown, HoverTool, TableColumn, DataTable, NumeralTickFormatter, Range1d, \
    LinearAxis, Legend, CustomJS, Slider, Column, LinearColorMapper, ColorBar, BasicTicker, Row, Select
from bokeh.models import Panel, Tabs, DateFormatter, Span
from bokeh.models.widgets import CheckboxButtonGroup
from bokeh.layouts import column, row, layout, gridplot
from bokeh.palettes import Spectral, Category20c, Dark2_5, Greys256, Magma256, Dark2_8, Colorblind
from bokeh.transform import factor_cmap, transform

def table_nonedit_general(df: pd.DataFrame, index_col: str = None, dt_columns:List[str] = None, index_position: int=None):
    df = df.copy()
    if index_col is not None:
        df = df.set_index(index_col)
    dt_columns = [] if dt_columns is None else dt_columns
    for col in dt_columns:
        df[col] = pd.to_datetime(df[col])
    source = ColumnDataSource(df)
    columns = [TableColumn(field=col, title = col, formatter = DateFormatter()) for col in dt_columns]
    columns.extend([TableColumn(field=col, title=col) for col in df.columns if col not in dt_columns])
    data_table = DataTable(source=source, columns=columns, editable=False,
                           index_position=index_position, sizing_mode="stretch_both")
    return data_table

def chart_coef_fimpt(df: pd.DataFrame, title: str, size: tuple = (1000, 500)) -> Figure:

    """Coefficient or Feature importance chart

    :param df: 1 row per feature , columns: [Features, Coef], optional column [Color]
    :param title: chart title
    :param size: figsize
    :return:
    """
    colors = cycle(Dark2_5)
    if 'Color' not in df.columns:
        df = df.copy()
        df.loc[:, 'Color'] = [c for _, c in zip(range(len(df)), colors)]

    # df[Features, Coef, Color]
    source_coef = ColumnDataSource(df)
    fig_coef = figure(plot_width = size[0], plot_height = size[1],
                      x_range = source_coef.data['Features'], title=title)
    fig_coef.vbar(
        source = source_coef, x = 'Features', top = 'Coef', bottom = 0,
        width = 0.5, line_color = 'black', fill_color = 'Color'
    )
    fig_coef.add_tools(HoverTool(tooltips=[("Features", "@Features"), ("Coef", "@Coef")]))
    fig_coef.xaxis.major_label_orientation = "vertical"
    return fig_coef

def table_hypt_distr(study_info: dict):
    """Hyperparameter distribution table chart

    :param study_info: dict, {
            'best_params': study.best_params, # dict
            'best_value': study.best_value, # float
            'param_distr': r_distr, # list of dict
            'trial_valus': trial_results, # list
            'hypt_importance': importance_
        }
    :return:
    """
    s = pd.DataFrame.from_records(study_info['param_distr'])
    s['Best'] = s['Hypt'].map(study_info['best_params'])
    s['Importance'] = s['Hypt'].map(study_info['hypt_importance'])
    head_cols = ['Hypt', 'Best', 'Importance', 'Distribution']
    s = s[head_cols + s.columns.difference(head_cols).tolist()].sort_values(by = 'Importance', ascending = False)

    data_table = table_nonedit_general(df = s, index_col='Hypt', index_position=None)
    return data_table


def chart_hypt_imp(hypt_importance: dict, title: str='Hyperparameter Importance'):
    """Hyperparameter importance Chart

    :param hypt_importance: dict, apis.get_hypt_info()['hypt_importance']
        {
            hyper_param1: v1,
            hyper_param2: v2,
        }
    :param title: chart title
    :return:
    """
    df = pd.DataFrame(hypt_importance.items())
    df.columns = ['Hypt', 'Importance']
    source_hypt = ColumnDataSource(df.sort_values("Importance"))
    fig_hypt = figure(plot_width=500, plot_height=500, y_range=source_hypt.data['Hypt'],
                      title=title)
    fig_hypt.hbar(
        source=source_hypt, y='Hypt', right='Importance',
        width=0.4, left=0,
        line_color='black',
        #fill_color='Color'
    )
    fig_hypt.add_tools(HoverTool(tooltips=[("Hypt", "@Hypt"), ("Importance", "@Importance")]))
    return fig_hypt


def chart_optbin(bin_r: pd.DataFrame, metric: str = 'Event rate', size: tuple = (1000, 500), title: str = "Realized Event vs. Bins") -> Figure:

    """Optimal binning result chart

    :param bin_r: results returned by optbin.binning_table.build(add_totals=False)
                    columns [Bin, Count, Count (%), Event, Non-event, Event rate, WoE, IV, JS]
    :param metric: metric to be displayed on right axis, default Event rate, can also be WoE, IV, JS
    :param size: figsize
    :return:
    """

    source_bin = ColumnDataSource(bin_r)
    fig_bin = figure(plot_width=size[0], plot_height=size[1], x_range=source_bin.data['Bin'], title = title)
    fig_bin.add_layout(Legend(), 'right')
    fig_bin.vbar_stack(
        ['Non-event', 'Event'], legend_label=['Non-event', 'Event'],
        source = source_bin, x = 'Bin', width = 0.8,
        line_color = 'black', color = ['#6EA3C6', '#CB4932']
    )
    fig_bin.yaxis.formatter = NumeralTickFormatter(format='0.0a')  # million
    # add event rate to secondary axis
    #fig_bin.y_range = Range1d(start = 0, end = 1)
    fig_bin.extra_y_ranges = {"BAD_RT_AXIS": Range1d(start=0, end = 1.2)}
    fig_bin.add_layout(LinearAxis(y_range_name="BAD_RT_AXIS"), 'right')
    fig_bin.line(x='Bin', y=metric, source=source_bin, y_range_name='BAD_RT_AXIS', width=3,
                    legend_label=metric, color='#062335')
    fig_bin.circle(x='Bin', y=metric, source=source_bin, y_range_name='BAD_RT_AXIS', color='black',
                      fill_color="white", size=10)
    fig_bin.add_tools(HoverTool(tooltips=[("Bin", "@Bin"), ("Count", "@Count"), ("Count (%)", "@{Count (%)}{(0.000)}"),
                                          ("Event", "@Event"), ("Non-event", "@{Non-event}"), ("Event rate", "@{Event rate}{(0.000)}"),
                                          ("WoE", "@WoE"), ("IV", "@IV"), ("JS", "@JS")]))
    fig_bin.xaxis.major_label_orientation = "vertical"

    return fig_bin


def chart_roc_curve(*metric_dfs: pd.DataFrame, names:List[str] = None, 
            size: tuple = (1000, 500), title ='ROC Curve') -> layout:
    """ROC curve on different models for comparison

    :param metric_dfs: metrics at each threshold, columns: [threshold, tpr, fpr]
    :param names: names for each metric dfs, use as labels in the chart to differientiate each curve
    :param size: figsize
    :param title: title of the chart
    :return: ROC curve with a slider for threshold
    """
    colors = cycle(Dark2_5)
    fig_roc = figure(plot_width=size[0], plot_height=size[1], title=title,
                     x_axis_label='FPR', y_axis_label='TPR')
    fig_roc.add_layout(Legend(), 'right')
    for metric_df, name in zip(metric_dfs, names):
        source_roc = ColumnDataSource(metric_df)
        fig_roc.line("fpr", "tpr", source = source_roc, color=next(colors), width=3, legend_label=name)
    fig_roc.add_tools(HoverTool(tooltips=[("threshold", "@threshold"), ("fpr", "@fpr"), ("tpr", "@tpr")]))

    with open(os.path.join(os.path.dirname(__file__), "chart_api_js/roc_standalone_callback.js"), "r") as obj:
        callback_ = obj.read()
        
    slider = Slider(start=0.0, end=1.0, value=1, step=0.0001, 
        title='Threshold', format='0[.]0000')
    for slider_pos, slider_df in enumerate(metric_dfs):
        # will add a slider to the relevant curve
        df_anno = pd.DataFrame({
            'threshold': [1], 
            'tpr': [0], 
            'fpr' : [0]
        })
        source_anno = ColumnDataSource(df_anno)
        fig_roc.circle("fpr", "tpr", source = source_anno, 
            color='black', fill_color="#2B699D", size = 10)
       
        callback = CustomJS(
            args=dict(
                source_roc_test=ColumnDataSource(slider_df), 
                source_anno=source_anno
            ),
            code=callback_
        )
        slider.js_on_change('value', callback)
    
    layout = Column(slider, fig_roc)
    return layout


def chart_pr_curve(*metric_dfs: pd.DataFrame, names:List[str] = None, 
        size: tuple = (1000, 500), title ='PR Curve') -> layout:
    """Precision - Recall curve on different models for comparison

    :param metric_dfs: metrics at each threshold, columns: [threshold, tpr, fpr]
    :param names: names for each metric dfs, use as labels in the chart to differientiate each curve
    :param size: figsize
    :param title: title of the chart
    :return: PR curve with a slider for threshold
    """
    colors = cycle(Dark2_8);next(colors);next(colors)
    fig_pr = figure(plot_width=size[0], plot_height=size[1], title=title,
                     x_axis_label='Recall', y_axis_label='Precision')
    fig_pr.add_layout(Legend(), 'right')
    for metric_df, name in zip(metric_dfs, names):
        source_roc = ColumnDataSource(metric_df)
        fig_pr.line("recall", "precision", source=source_roc, color=next(colors), width=3, legend_label=name)
    fig_pr.add_tools(HoverTool(tooltips=[("threshold", "@threshold"), ("recall", "@recall"), ("precision", "@precision")]))

    with open(os.path.join(os.path.dirname(__file__), "chart_api_js/pr_standalone_callback.js"), "r") as obj:
        callback_ = obj.read()
    
    slider = Slider(start=0.0, end=1.0, value=1, step=0.0001, 
            title='Threshold', format='0[.]0000')
    
    for slider_pos, slider_df in enumerate(metric_dfs):
        # will add a slider to the relevant curve
        df_anno = pd.DataFrame({
            'threshold': [1], 
            'recall': [0], 
            'precision' : [1]
        })
        source_anno = ColumnDataSource(df_anno)
        fig_pr.circle("recall", "precision", source=source_anno, 
            color='black', fill_color="#2B699D", size=10)

        callback = CustomJS(
            args=dict(
                source_pr_test=ColumnDataSource(slider_df), 
                source_anno=source_anno
            ),
            code=callback_
        )
        
        slider.js_on_change('value', callback)
    
    layout = Column(slider, fig_pr)
    return layout


def chart_pr_threshold(metrics: pd.DataFrame, size: tuple = (1000, 500), title: str ='PR Tradeoff Curve') -> Figure:
    """Precision Recall tradeoff curve @ each different threshold

    :param metrics: metrics at each threshold, columns: [threshold, precision, recall]
    :param size: figsize
    :param title: title of the chart
    :return:
    """
    source_pr = ColumnDataSource(metrics)
    fig_pr = figure(plot_width=size[0], plot_height=size[1], title=title,
                    x_axis_label='Threshold', y_axis_label='Precision/Recall')
    fig_pr.add_layout(Legend(), 'right')
    fig_pr.line("threshold", "recall", source=source_pr, color='#2EA496', width=3, legend_label='Recall')
    fig_pr.line("threshold", "precision", source=source_pr, color='#CB6040', width=3, legend_label='Precision')
    fig_pr.add_tools(HoverTool(tooltips=[("threshold", "@threshold"), ("recall", "@recall"), ("precision", "@precision")]))
    return fig_pr


def chart_confusion_maxtrix(metrics: pd.DataFrame, size: tuple = (1000, 500), title: str ='Confusion Matrix') -> layout:
    """Confusion matrix at each threshold (Count + Error chart)

    :param metrics: metrics at each threshold, columns: [threshold, fp, tp, tn, fn]
    :param size: figsize
    :param title: title of the chart
    :return: Confusion matrix (count view + error view) and slider for threshold
    """
    initial = metrics.iloc[len(metrics) // 2, :]
    total = metrics.loc[0, ['fp', 'tp', 'tn', 'fn']].sum()

    df_origin = pd.DataFrame(
        [[initial['fp'], initial['tp']], [initial['tn'], initial['fn']]],
        columns=['neg', 'pos'],
        index=['pos', 'neg']
    )
    df_origin.index.name = 'Prediction'
    df_origin.columns.name = 'Actual'
    # Prepare data.frame in the right format
    df_error = pd.DataFrame(
        [[initial['fpr'], 0], [0, 1 - initial['tpr']]],
        columns=['neg', 'pos'],
        index=['pos', 'neg']
    )
    df_error.index.name = 'Prediction'
    df_error.columns.name = 'Actual'
    df_origin = df_origin.stack().rename("value").reset_index()
    df_error = df_error.stack().rename("value").reset_index()
    # You can use your own palette here

    # colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
    origin_source = ColumnDataSource(df_origin)
    error_source = ColumnDataSource(df_error)
    raw_source = ColumnDataSource(metrics)

    # Had a specific mapper to map color with value
    mapper_origin = LinearColorMapper(palette=Magma256, low=0, high=total)
    mapper_error = LinearColorMapper(palette=Greys256, low=0.0, high=1.0)

    # Define a figure

    p_origin = figure(
        plot_width=size[0] // 2, plot_height=size[1], title=f"{title} - Original",
        x_range=['neg', 'pos'], y_range=['pos', 'neg']
    )

    # Create rectangle for heatmap
    p_origin.rect(
        x="Prediction", y="Actual", source=origin_source,
        width=1, height=1, line_color=None, fill_color=transform('value', mapper_origin)
    )

    # Add legend
    color_bar_origin = ColorBar(
        color_mapper=mapper_origin,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=256)
    )
    p_origin.add_layout(color_bar_origin, 'right')
    p_origin.add_tools(HoverTool(tooltips=[("Prediction", "@Prediction"), ("Actual", "@Actual"), ("value", "@value")]))


    p_error = figure(
        plot_width=size[0] // 2, plot_height=size[1], title=f"{title} - Error",
        x_range=['neg', 'pos'], y_range=['pos', 'neg']
    )

    # Create rectangle for heatmap
    p_error.rect(
        x="Prediction", y="Actual", source=error_source,
        width=1, height=1, line_color=None, fill_color=transform('value', mapper_error)
    )

    # Add legend
    color_bar_error = ColorBar(
        color_mapper=mapper_error,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=256)
    )
    p_error.add_layout(color_bar_error, 'right')
    p_error.add_tools(HoverTool(tooltips=[("Prediction", "@Prediction"), ("Actual", "@Actual"), ("value", "@value")]))

    slider = Slider(start=0.0, end=1.0, value=initial['threshold'], step=0.001, title='Threshold', format = '0[.]000')

    with open(os.path.join(os.path.dirname(__file__), "chart_api_js/confunsion_matrix_callback.js"), "r") as obj:
        callback = obj.read()
    callback = CustomJS(
        args=dict(origin_source=origin_source, error_source=error_source, raw_source=raw_source),
        code=callback
    )
    slider.js_on_change('value', callback)

    layout = Column(slider, Row(p_origin, p_error))
    return layout



def chart_metric_score_compare(scores: dict, size: tuple = (1000, 400), title: str ='Metric by model') -> Figure:
    """metric bar chart of each model/submodel

    :param scores: dictionary of metrics, keys are model names, values are corresponding metric
    :param size: figsize
    :param title: chart title
    :return:

    """
    colors = cycle(Dark2_8);next(colors);next(colors);next(colors)
    df = pd.DataFrame({
        'Model': scores.keys(), 
        'Metric': scores.values(), 
        'Color': [c for _, c in zip(scores.keys(), colors)]
    })

    #df.loc[df['Metric'] < df['Metric'].quantile(0.3), 'Color'] = '#DE5644'

    source_metrics = ColumnDataSource(df)

    fig_metric = figure(plot_width=size[0], plot_height=size[1], title=title,
                        x_range=source_metrics.data['Model'])
    fig_metric.add_layout(Legend(), 'right')
    fig_metric.vbar(
        source=source_metrics, x='Model', top='Metric',
        width=0.5, bottom=0,
        line_color='black', fill_color='Color', legend_field="Model"
    )
    fig_metric.add_tools(HoverTool(tooltips=[("Model", "@Model"), ("Metric", "@Metric")]))
    #fig_metric.xaxis.major_label_orientation = 3.14/4
    fig_metric.xaxis.visible = False
    return fig_metric

def chart_calibration_curve(cali_curves: Dict[str, Dict[str, list]], size: tuple = (1200, 400), title: str ='Calibration Curve') -> Figure:
    """plot calibration curve for multiple models

    :param cali_curves: assemble of results from sklearn.calibration_curve
        format cali_curves = {
            'cali1' : {
                # from calibration_curve
                'prob_true': [],
                'prob_pred': [],
            },
            'cali2': {
                # from calibration_curve
                'prob_true': [],
                'prob_pred': [],
            },
            ...
        }
    :param size:
    :param title:
    :return:

    """
    colors = cycle(Dark2_5)
    fig_cali = figure(plot_width=size[0], plot_height=size[1], title=title,
                      x_axis_label='Mean predicted probability', y_axis_label='Fraction of Positives')
    fig_cali.add_layout(Legend(), 'right')
    fig_cali.line(x=[0, 1], y=[0, 1], color='#7A7C77', width=2, line_dash="4 4", legend_label='Perfect Calibrate')
    for cali_name, cali_curve in cali_curves.items():
        source_cali =  ColumnDataSource(cali_curve)
        fig_cali.line("prob_pred", "prob_true", source=source_cali, color=next(colors), width=3, legend_label=cali_name)
        fig_cali.circle("prob_pred", "prob_true", source=source_cali, color='black', fill_color="white", size=10)

    fig_cali.add_tools(HoverTool(tooltips=[("prob_pred", "@prob_pred"), ("prob_true", "@prob_true")]))
    return fig_cali

