import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import cycle
from bokeh.plotting import figure, show, Figure
from bokeh.models import ColumnDataSource, Dropdown, HoverTool, TableColumn, DataTable, NumeralTickFormatter, Range1d, \
    LinearAxis, Legend, CustomJS, Slider, ScientificFormatter, StringFormatter, LinearColorMapper, ColorBar, BasicTicker
from bokeh.models import Panel, Tabs, DateFormatter, Span, LegendItem, Legend, Div, Panel, Tabs
from bokeh.layouts import column, row, layout, gridplot
from bokeh.palettes import Dark2_5, Dark2_8, Magma256
from bokeh.transform import factor_cmap, transform
from bokeh.io import save, output_file
from ModelingTools.Explore.profiling import StatVar, QuantileStat, DescStat, NumericStat, CategStat, TabularStat, \
        CategUniVarClfTargetCorr, NumericUniVarClfTargetCorr, TabularUniVarClfTargetCorr

def _sort_combine(arr: pd.Series, max_item:int = 10) -> pd.Series:
    arr = arr.sort_values(ascending = False)
    if len(arr) > max_item:
        arr.iloc[max_item] = arr.iloc[max_item :].sum()
        arr = arr.iloc[:max_item + 1]
        arr.index = arr.index[:-1].tolist() + ['Other']

    return arr


def chart_donut(donut_arr: pd.Series, max_donut:int = 6, colors: list = None, size: tuple = (500, 400), title: str='Donut chart') -> Figure:
    if colors is None:
        donut_arr = _sort_combine(donut_arr, max_item = max_donut)
    perc = donut_arr / donut_arr.sum()
    angles = perc.map(lambda x: 2 * 3.1415926 * x).cumsum()
    color = cycle(Dark2_5)
    donut_dict = pd.DataFrame({
        'label' : donut_arr.index,
        'value' : donut_arr.values,
        'perc' : perc,
        'start' : [0] + angles[:-1].values.tolist(),
        'end' : angles.values,
    })
    if colors is not None:
        donut_dict['colors'] = colors
    else:
        donut_dict['colors'] = donut_dict['label'].map(lambda x: next(color) if x != 'Other' else '#9C9C9C')
    
    source = ColumnDataSource(donut_dict)
    p = figure(plot_width=size[0], plot_height=size[1], title=title)
    p.add_layout(Legend(), 'right')
    p.annular_wedge(
        x=0, y=0, source = source,
        inner_radius=0.5, outer_radius=0.85,
        start_angle="start", end_angle="end", legend_group = 'label',
        line_color="black", line_width=2, fill_color='colors'
    )
    p.x_range = Range1d(start=-1, end = 1)
    p.y_range = Range1d(start=-1, end = 1)
    p.add_tools(HoverTool(tooltips=[("label", "@label"), ("value", "@value"), ("perc", "@perc{(0.0%)}")]))
    p.axis.visible = False
    p.grid.visible = False
    return p

def chart_barchart(bar_arr: pd.Series, max_bar:int = 10, size: tuple = (500, 400), title: str='Bar chart') -> Figure:
    
    bar_arr = _sort_combine(bar_arr, max_item = max_bar).sort_values()
    perc = bar_arr / bar_arr.sum()
    color = cycle(Dark2_8)
    bar_dict = pd.DataFrame({
        'label' : bar_arr.index.astype('string').tolist(),
        'value' : bar_arr.values,
        'perc' : perc
    })
    bar_dict['colors'] = bar_dict['label'].map(lambda x: next(color) if x != 'Other' else '#9C9C9C')
    
    source = ColumnDataSource(bar_dict)
    p = figure(plot_width=size[0], plot_height=size[1], title=title,
            y_range = source.data['label'])
    p.add_layout(Legend(), 'right')
    p.hbar(
        y='label', right='value', left = 0, source = source,
        height=0.75, legend_group = 'label',
        line_color="black", line_width=2, fill_color='colors'
    )
    p.add_tools(HoverTool(tooltips=[("label", "@label"), ("value", "@value"), ("perc", "@perc{(0.0%)}")]))
    #p.yaxis.major_label_orientation = "vertical"
    p.xaxis.formatter = NumeralTickFormatter(format='0.0a')  # million
    return p

def chart_histogram(bins_edges: np.ndarray, hists: np.ndarray, quantile_stat: QuantileStat = None, desc_stat: DescStat = None, size: tuple = (1200, 500), title: str ='Distribution chart') -> Figure:
    distr_bins = pd.DataFrame({
        'lower' : bins_edges[:-1],
        'upper' : bins_edges[1:],
        'counts' : hists,
    })
    distr_bins['percent'] = distr_bins['counts'] / distr_bins['counts'].sum()
    max_height = hists.max() * 1.2

    source_hist_cali = ColumnDataSource(distr_bins)
    
    p = figure(plot_width=size[0], plot_height=size[1], title=title,
        x_axis_label='Value', y_axis_label='Counts')
    
    # optional hareas
    if desc_stat is not None:
        p.harea(y=[0, max_height], x1 = desc_stat.mean - 3 * desc_stat.std, x2 = desc_stat.mean + 3 * desc_stat.std,
                fill_color="#0687AD", alpha = 0.2, legend_label="± 3σ")
        
    if quantile_stat is not None:
        p.harea(y=[0, max_height], x1 = quantile_stat.perc_1th, x2 = quantile_stat.perc_99th,
                fill_color="#06AD1A", alpha = 0.2, legend_label="Perc 1-99")

        p.harea(y=[0, max_height], x1 = quantile_stat.q3 - 1.5 * quantile_stat.iqr, x2 = quantile_stat.q3 + 1.5 * quantile_stat.iqr,
                fill_color="#E05907", alpha = 0.2, legend_label="IQR")

    # histogram
    p.quad(source=source_hist_cali, bottom=0, top='counts',
        left='lower', right='upper',
        fill_color='#465E83', line_color='black'
    )
        
    # optional vlines
    if quantile_stat is not None:
        p.line(
            x = [quantile_stat.median, quantile_stat.median], y = [0, max_height], 
            legend_label = 'median', line_color='#06AD1A', line_width = 2, line_dash = 'dashed'
        )
    if desc_stat is not None:
        p.line(
            x = [desc_stat.mean, desc_stat.mean], y = [0, max_height], 
            legend_label = 'mean', line_color='#07B2E0', line_width = 2, line_dash = 'dashed'
        )
        
        
    p.y_range = Range1d(start=0, end = max_height)
    p.yaxis.formatter = NumeralTickFormatter(format='0.0a')  # million
    p.add_tools(
        HoverTool(tooltips=[("Range", "@lower to @upper"), ("Count", "@counts"), ("Percent", "@percent{(0.00%)}")]))
    return p

def chart_categ_distr(vcounts: pd.Series, size: tuple = (1200, 600), title: str ='Distribution chart') -> Figure:
    fig = chart_barchart(
        vcounts,
        max_bar = 20,
        size = size,
        title = title
    )

    return fig


def metric_div(value: float, perc: float, label:str, threshold_p: float = 1.0, back_color: str = 'white') -> Div:
    html = """
    <style>
        h1 {text-align: center;}
        p {text-align: center}
        div {text-align: center;}

        .value {
            text-align: center;
            font-size: 20px;
            color: rgb(44, 144, 129);
            line-height:5px;
        }
        .perc {
            text-align: center;
            font-size: 15px;
            color: rgb(163, 113, 33);
            line-height:5px;
        }
        .label {
            text-align: center;
            font-size: 30px;
            color: rgb(1, 1, 1);
            line-height:25px;
        }
    </style>
    """
    style = "color:red" if perc > threshold_p else ""
    html +=  f"""
    <p class="value">{value:,.0f}</p>
    <p class="perc">({perc:.1%})</p>
    <p class="label" style="{style}">{label}</p>
    """
    return Div(
        text = html, 
        background = back_color, 
        align = 'center', 
        min_width = 200,
        margin = (2, 2, 2, 2),
        width_policy = 'max',
    )

def numeric_count_summary(total: int, missing: StatVar, zeros: StatVar, inf_pos: StatVar, inf_neg: StatVar, xtreme: StatVar) -> layout:
    valid = total - missing.value - zeros.value - inf_pos.value - inf_neg.value - xtreme.value
    valid_perc = 1 - missing.perc - zeros.perc - inf_pos.perc - inf_neg.perc - xtreme.perc

    rows = column([
            row(
                [
                    metric_div(valid, valid_perc, 'Valid', threshold_p = 1, back_color = 'ivory'),
                    metric_div(missing.value, missing.perc, 'Missing', threshold_p = 0.1, back_color = 'gainsboro'),
                    metric_div(zeros.value, zeros.perc, 'Zeros', threshold_p = 0.1, back_color = 'azure'),
                ],  
                sizing_mode='stretch_both'
            ),
            row(
                [
                    metric_div(inf_pos.value, inf_pos.perc, '+Inf', threshold_p = 0.01, back_color = 'lightpink'),
                    metric_div(inf_neg.value, inf_neg.perc, '-Inf', threshold_p = 0.01, back_color = 'lightgreen'),
                    metric_div(xtreme.value, xtreme.perc, 'Xtreme', threshold_p = 0.05, back_color = 'peachpuff')
                ],  
                sizing_mode='stretch_both'
            )
        ],
        sizing_mode='stretch_both'
    )
    return rows

def categ_count_summary(total: int, missing: StatVar, unique: StatVar) -> layout:
    valid = total - missing.value
    valid_perc = 1 - missing.perc
    rows = row(
        [
            metric_div(valid, valid_perc, 'Valid', threshold_p = 1, back_color = 'ivory'),
            metric_div(missing.value, missing.perc, 'Missing', threshold_p = 0.1, back_color = 'gainsboro'),
            metric_div(unique.value, unique.perc, 'Unique', back_color = 'aquamarine'),
        ],  
        sizing_mode='stretch_both'
    )
    return rows

def numeric_donut(total: int, missing: StatVar, zeros: StatVar, inf_pos: StatVar, inf_neg: StatVar, xtreme: StatVar,
            size: tuple = (600, 500), title: str = 'Composite Donut') -> Figure:
    valid = total - missing.value - zeros.value - inf_pos.value - inf_neg.value - xtreme.value


    donut_dict = pd.Series(
        [valid, missing.value, zeros.value, inf_pos.value, inf_neg.value,  xtreme.value],
        index = ['Valid', 'Missing', 'Zeros', '+Inf', '-Inf', 'Xtreme'],
    )

    fig = chart_donut(
        donut_dict,
        colors = ['gold', 'darkgray', 'azure', 'mediumvioletred', 'olivedrab', 'tomato'],
        size = size,
        title = title
    )
    return fig

def categ_donut(total: int, missing: StatVar, size: tuple = (600, 500), title: str = 'Composite Donut') -> Figure:
    valid = total - missing.value
    donut_dict = pd.Series(
        [valid, missing.value],
        index = ['Valid', 'Missing'],
    )

    fig = chart_donut(
        donut_dict,
        colors = ['gold', 'darkgray'],
        size = size,
        title = title
    )
    return fig
        


def table_stat(stats: dict) -> DataTable:
    df = (
        pd.Series(stats)
        .reset_index(name='VALUE')
        .rename(columns = {'index' : 'METRIC'})
    )
    source = ColumnDataSource(df)
    data_table = DataTable(
        source=source, 
        columns=[
            TableColumn(
                field = 'METRIC',
                title = 'METRIC',
                sortable = False,
                formatter = StringFormatter(
                    font_style = 'bold'
                )
            ),
            TableColumn(
                field = 'VALUE',
                title = 'VALUE',
                sortable = False,
                formatter = ScientificFormatter(
                    precision = 5,
                    text_color = 'darkslategrey',
                )
            )
        ], 
        editable=False,
        index_position = None,
        height = None,
        sizing_mode = 'stretch_both'
    )
    return data_table

def chart_boxplot(p_x_y: pd.DataFrame, xname:str = 'x', yname:str = 'y', 
        size: tuple = (500, 400), title: str='Boxplot') -> Figure:
    
    p_x_y['y'] = p_x_y.index.astype(str)
    data_source = ColumnDataSource(p_x_y)

    p = figure(plot_width=size[0], plot_height=size[1],
            title=title, x_range=data_source.data['y'],
            x_axis_label=yname, y_axis_label=xname)


    # boxes
    p.vbar(x='y', width=0.7, bottom='median', top='q3', 
        source=data_source, fill_color="#E08E79", line_color="black")
    p.vbar(x='y', width=0.7, bottom='q1', top='median', 
        source=data_source, fill_color="#3B8686", line_color="black")
    # whiskers (almost-0 height rects simpler than segments)
    p.rect(x='y', y='rbound', width = 0.2, height = 0.005, 
        source=data_source, line_color="black")
    p.rect(x='y', y='lbound', width = 0.2, height = 0.005, 
        source=data_source, line_color="black")
    p.rect(x='y', y='mean', width = 0.7, height = 0.005, 
        source=data_source, line_color="#B63B3B", fill_color="#B63B3B",
        legend_label='mean')
    # stems
    p.segment(x0='y', y0='rbound', x1='y', y1='q3', 
        source=data_source, line_color="black",
    )
    p.segment(x0='y', y0='lbound', x1='y', y1='q1', 
        source=data_source, line_color="black",
    )

    p.add_tools(HoverTool(tooltips=[
        ("label", "@y"), ("mean", "@mean"), ("lbound", "@lbound"), ("q1", "@q1"),
        ("median", "@median"), ("q3", "@q3"), ("rbound", "@rbound")
    ]))

    p.xgrid.grid_line_color = None

    #p.yaxis.major_label_orientation = "vertical"
    p.yaxis.formatter = NumeralTickFormatter(format='0.0a')  # million
    
    return p

def chart_optbin_multiclass(bin_r: pd.DataFrame, ylabels: List[str] = None, xname:str = 'x', yname:str = 'y',
        size: tuple = (1000, 500), title: str = "Realized Event vs. Bins") -> Figure:

    """Optimal binning result chart

    :param bin_r: results returned by optbin.binning_table.build(add_totals=False)
                    columns [Bin, Count, Count (%), Event_x, Event rate_y]
    :param ylabels: if given, use this list to identify the columns
    :param size: figsize
    :return:
    """
    # extract target labels
    if ylabels is None:
        ylabels = bin_r.columns[bin_r.columns.str.startswith('Event_rate_')].str.replace('Event_rate_', '').tolist()
    
    color = cycle(Dark2_8)
    colors = [next(color) for _ in ylabels]
    
    source_bin = ColumnDataSource(bin_r)

    fig_bin = figure(plot_width=size[0], plot_height=size[1],
        x_axis_label = xname, y_axis_label = yname,
        x_range=source_bin.data['Bin'], title = title)

    fig_bin.add_layout(Legend(), 'right')
    fig_bin.vbar_stack(
        [f"Event_{y_label}" for y_label in ylabels], # ['Event_0', 'Event_1'] 
        legend_label = ylabels,
        source = source_bin, x = 'Bin', width = 0.8,
        line_color = 'black', 
        color = colors
    )

    fig_bin.yaxis.formatter = NumeralTickFormatter(format='0.0a')  # million

    # add event rate to secondary axis
    #fig_bin.y_range = Range1d(start = 0, end = 1)
    fig_bin.extra_y_ranges = {"EVENT_RT_AXIS": Range1d(start=0, end = 1.2)}
    fig_bin.add_layout(LinearAxis(y_range_name="EVENT_RT_AXIS"), 'right')
    
    for i, ylabel in enumerate(ylabels):
        fig_bin.line(x='Bin', y=f'Event_rate_{ylabel}', source=source_bin, 
            y_range_name='EVENT_RT_AXIS', width=3,
            legend_label=f'Event_rate_{ylabel}', color=colors[i]
        )
        fig_bin.circle(x='Bin', y=f'Event_rate_{ylabel}', source=source_bin, 
            y_range_name='EVENT_RT_AXIS', color='black',
            fill_color="white", size=10
        )
    
    fig_bin.add_tools(HoverTool(tooltips=[
        ("Bin", "@Bin"), ("Count", "@Count"), ("Count (%)", "@{Count (%)}{(0.000)}")] 
        + [(ylabel, "@{Event_" + ylabel + "}") for ylabel in ylabels] 
        + [(f"Event_rate_{ylabel}", "@{Event_rate_" + ylabel + "}") for ylabel in ylabels]
    ))
    fig_bin.xaxis.major_label_orientation = "vertical"
    fig_bin.y_range = Range1d(start = 0, end = bin_r['Count'].max() * 1.2)

    return fig_bin

def chart_segment_group_count(seg_df: Dict[str, pd.DataFrame], group_name:str = 'y', agg_name:str = 'x', 
        size: tuple = (500, 500), title: str = "Stacking Segment Count by Group") -> Figure:
    
    seg = pd.concat([s['count'].rename(group) for group, s in seg_df.items()], axis = 1).T
    seg.columns = seg.columns.astype('str')
    aggs = seg.columns.astype('str').tolist()
    seg[group_name] = seg.index.astype(str)
    
    color = cycle(Dark2_8)
    colors = [next(color) for _ in aggs]
    
    source_bin = ColumnDataSource(seg)

    p = figure(plot_width=size[0], plot_height=size[1],
        x_axis_label = group_name, y_axis_label = agg_name,
        x_range=source_bin.data[group_name], title = title)

    p.add_layout(Legend(), 'right')
    p.vbar_stack(
        aggs,
        x = group_name,
        source = source_bin,
        legend_label = aggs,
        width = 0.8,
        line_color = 'black', 
        color = colors
    )

    p.yaxis.formatter = NumeralTickFormatter(format='0.0a')  # million
    
    p.add_tools(HoverTool(tooltips=[
        ("group", f"@{group_name}")]
        + [(agglabel, "@{" + agglabel + "}") for agglabel in aggs] 
    ))
    
    return p

def chart_segment_group_perc(seg_df: Dict[str, pd.DataFrame], group_name:str = 'y', agg_name:str = 'x', 
        size: tuple = (500, 500), title: str = "Segment Distribution by Group") -> Figure:
    
    seg = pd.concat([s['perc'].rename(group) for group, s in seg_df.items()], axis = 1).T
    seg.columns = seg.columns.astype('str')
    aggs = seg.columns.astype('str').tolist()
    seg[group_name] = seg.index.astype(str)
    
    color = cycle(Dark2_8)
    colors = [next(color) for _ in aggs]
    
    source_bin = ColumnDataSource(seg)

    p = figure(plot_width=size[0], plot_height=size[1],
        x_axis_label = group_name, y_axis_label = agg_name,
        x_range=source_bin.data[group_name], title = title)

    p.add_layout(Legend(), 'right')
    
    for i, agg in enumerate(aggs):
        p.line(x=group_name, y=agg, source=source_bin, 
            width=3, legend_label=agg, color=colors[i]
        )
        p.circle(x=group_name, y=agg, source=source_bin, 
            color='black', fill_color="white", size=10
        )

    p.yaxis.formatter = NumeralTickFormatter(format='0.00%')  # million
    
    p.add_tools(HoverTool(tooltips=[
        ("group", f"@{group_name}")]
        + [(agglabel, "@{" + agglabel +"}{(0.00%)}") for agglabel in aggs] 
    ))
    
    return p

def plot_numeric(nst: NumericStat, html_path: str = None) -> column:
    summary = numeric_count_summary(
        nst.total_, 
        nst.missing_, 
        nst.zeros_, 
        nst.infs_pos_, 
        nst.infs_neg_, 
        nst.xtreme_
    )
    donut = numeric_donut(
        nst.total_, 
        nst.missing_, 
        nst.zeros_, 
        nst.infs_pos_, 
        nst.infs_neg_, 
        nst.xtreme_,
        size = (600, 500)
    )

    tabs_desc_stat = [Panel(
        child = table_stat(nst.stat_descriptive_._asdict()),
        title="Origin"
    )]
    tabs_quant_stat = [Panel(
        child = table_stat(nst.stat_quantile_._asdict()),
        title="Origin"
    )]
    if nst.log_scale_:
        tabs_desc_stat.append(Panel(
            child = table_stat(nst.stat_descriptive_log_._asdict()),
            title="Log"
        ))
        tabs_quant_stat.append(Panel(
            child = table_stat(nst.stat_quantile_log_._asdict()),
            title="Log"
        ))

    tabs_hist = [Panel(
        child = chart_histogram(
            bins_edges = nst.bin_edges_,
            hists = nst.hist_,
            quantile_stat=nst.stat_quantile_,
            desc_stat=nst.stat_descriptive_,
            title = 'Histogram for Valid Values',
            size = (1200, 500)
        ),
        title="Histogram - Origin"
    )]
    if nst.log_scale_:
        tabs_hist.append(Panel(
            child = chart_histogram(
                bins_edges = nst.bin_edges_log_,
                hists = nst.hist_log_,
                quantile_stat=nst.stat_quantile_log_,
                desc_stat=nst.stat_descriptive_log_,
                title = 'Histogram for Valid Values',
                size = (1200, 500)
            ),
            title="Statistics - Log"
        ))

    # assemble to widget
    table_widgets = [
        Div(text="""
            <h3 style="text-align:left">Quantile Statistics for Valid Values</h2>
        """),
        Tabs(tabs=tabs_quant_stat),
        Div(text="""
            <h3 style="text-align:left">Descriptive Statistics for Valid Values</h2>
        """),
        Tabs(tabs=tabs_desc_stat),
    ]

    if nst.xtreme_method_ is not None:
        tabs_xtreme = [Panel(
            child = table_stat(nst.xtreme_stat_._asdict()),
            title="Origin"
        )]
        if nst.log_scale_:
            tabs_xtreme.append(Panel(
                child = table_stat(nst.xtreme_stat_log_._asdict()),
                title="Log"
            ))
        table_widgets.extend([
            Div(text=f"""
                <h3>Xtreme Value Statistics</h2><p style="text-align:left">method = {nst.xtreme_method_}</p>
            """),
            Tabs(tabs=tabs_xtreme),
        ])

    result = column([
        row([
            column([
                summary, 
                donut,
            ],
            sizing_mode = 'stretch_width'
            ),
            column(
                table_widgets,
                sizing_mode = 'stretch_width'
            )
        ]),
        Tabs(tabs=tabs_hist),
    ],
    #sizing_mode = 'fixed'
    )
    
    if html_path:
        output_file(html_path)
        save(result)
        
    return result


def plot_categ(nst: CategStat, html_path: str = None) -> row:
    summary = categ_count_summary(
        nst.total_, 
        nst.missing_, 
        nst.unique_
    )
    donut = categ_donut(
        total=nst.total_,
        missing=nst.missing_,
        size = (600, 500)
    )
    distr = chart_categ_distr(
        vcounts = nst.vcounts_,
        size = (600, 640)
    )

    result = row([
        column([
            summary,
            donut
        ],
        sizing_mode = 'stretch_width'
        ),
        distr
    ],
    sizing_mode = 'stretch_width'
    )
    
    if html_path:
        output_file(html_path)
        save(result)
        
    return result

def plot_numeric_clf_target_corr(nuct: NumericUniVarClfTargetCorr, html_path: str = None) -> row:
    
    tabs_x_y = Tabs(
        tabs = [
            Panel(
                child = chart_boxplot(
                    nuct.p_x_y_['origin'],
                    xname=nuct.colname_, 
                    yname=nuct.yname_,
                    size=(400, 500),
                    title = "P(x|y) - Feature Distribution By Target"
                ),
                title="Origin"
            ),
            Panel(
                child = chart_boxplot(
                    nuct.p_x_y_['log'],
                    xname=nuct.colname_, 
                    yname=nuct.yname_,
                    size=(400, 500),
                    title = "P(x|y) - Feature Distribution By Target"
                ),
                title="Log"
            )
        ]
    )
    
    fig_y_x = chart_optbin_multiclass(
        nuct.p_y_x_,
        ylabels = nuct.ylabels_,
        xname=nuct.colname_, 
        yname=nuct.yname_,
        size=(1000, 500),
        title = "P(y|x) - Event Rate by Bucketized Feature"
    )
    
    
    result = row([fig_y_x, tabs_x_y], 
        sizing_mode='stretch_height')
    
    if html_path:
        output_file(html_path)
        save(result)
    return result
    
    
def plot_categ_clf_target_corr(cuct: CategUniVarClfTargetCorr, html_path: str = None) -> row:
    tabs_x_y = Tabs(
        tabs = [
            Panel(
                child = chart_segment_group_count(
                    cuct.p_x_y_,
                    group_name=cuct.yname_, 
                    agg_name=cuct.colname_,
                    size=(600, 500),
                    title = "P(x|y) - Category Count By Target"
                ),
                title="Count"
            ),
            Panel(
                child = chart_segment_group_perc(
                    cuct.p_x_y_,
                    group_name=cuct.yname_, 
                    agg_name=cuct.colname_,
                    size=(600, 500),
                    title = "P(x|y) - Category Distribution By Target"
                ),
                title="Percentage"
            )
        ]
    )
    
    tabs_y_x = Tabs(
        tabs = [
            Panel(
                child = chart_segment_group_count(
                    cuct.p_y_x_,
                    group_name=cuct.colname_, 
                    agg_name=cuct.yname_,
                    size=(800, 500),
                    title = "P(y|x) - Event Count by Feature Categories"
                ),
                title="Count"
            ),
            Panel(
                child = chart_segment_group_perc(
                    cuct.p_y_x_,
                    group_name=cuct.colname_, 
                    agg_name=cuct.yname_,
                    size=(800, 500),
                    title = "P(x|y) - Event Rate by Feature Categories"
                ),
                title="Percentage"
            )
        ]
    )
    result = row([tabs_y_x, tabs_x_y], 
        sizing_mode='stretch_height')
    
    if html_path:
        output_file(html_path)
        save(result)
    return result

def table_summary(ts: TabularStat) -> row:
    num_numeric = len(list(filter(lambda c: isinstance(c, NumericStat), ts.configs.values())))
    broad_categs = list(filter(lambda c: isinstance(c, CategStat), ts.configs.values()))
    num_binary = len(list(filter(lambda c: c.binary_ is True, broad_categs)))
    num_categs = len(list(filter(lambda c: c.binary_ is False, broad_categs)))
    total = len(ts.configs)
    rows = row(
        [
            metric_div(num_numeric, num_numeric / total, 'Numerical Features', threshold_p = 1),
            metric_div(num_binary, num_binary / total, 'Binary Features', threshold_p = 1),
            metric_div(num_categs, num_categs / total, 'Categorical Features', threshold_p = 1),
        ],  
        sizing_mode='stretch_both'
    )
    return rows

def plot_table_profiling(ts: TabularStat, html_path: str = None) -> column:
    figs = [
        Div(
            text = """
                <h1>Overview</h1>
            """
        ),
        table_summary(ts)
    ]
    for col, config in ts.configs.items():
        if isinstance(config, NumericStat):
            fig = plot_numeric(config)
        elif isinstance(config, CategStat):
            fig = plot_categ(config)

        figs.append(
            Div(
                text = f"""
                    <h2 style="color:#395687">{col}</h2>
                """
            )
        )
        figs.append(fig)
    
    result = column(figs, sizing_mode = 'stretch_both')
    
    if html_path:
        output_file(html_path)
        save(result)
        
    return result

def plot_uni_clf_target_corr(tuvct: TabularUniVarClfTargetCorr, html_path: str = None) -> column:
    figs = [
        Div(
            text = """
                <h1>Univaraite Feature-Target Correlation for Classification</h1>
            """
        )
    ]
    for col, config in tuvct.configs.items():
        if isinstance(config, NumericUniVarClfTargetCorr):
            fig = plot_numeric_clf_target_corr(config)
        elif isinstance(config, CategUniVarClfTargetCorr):
            fig = plot_categ_clf_target_corr(config)

        figs.append(
            Div(
                text = f"""
                    <h2 style="color:#395687">{col}</h2>
                """
            )
        )
        figs.append(fig)
    
    result = column(figs, sizing_mode = 'stretch_both')
    
    if html_path:
        output_file(html_path)
        save(result)
        
    return result


####### Other plots

def chart_gridplot(grid: pd.DataFrame, reverse_color:bool = False, size: tuple = (1000, 600), title: str ='Grid Plot') -> layout:
    grid_flat = grid.stack().rename("value").reset_index()

    grid_flat_source = ColumnDataSource(grid_flat)
    mapper_color = LinearColorMapper(palette=Magma256[::-1] if reverse_color else Magma256)
    p = figure(
        plot_width=size[0], plot_height=size[1], title=title,
        x_axis_label=grid.columns.name, 
        y_axis_label=grid.index.name,
        x_range = grid.columns.tolist(), 
        y_range = grid.index.tolist()
    )
    p.rect(
        x=grid.columns.name,
        y=grid.index.name,
        width=1,
        height=1,
        source=grid_flat_source,
        line_color=None,
        fill_color=transform('value', mapper_color)
    )

    p.line(x = grid.index.tolist(), y = grid.columns.tolist(), 
           color='red', width=2, line_dash="4 4")

    # Add legend
    color_bar = ColorBar(
        color_mapper=mapper_color,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=256)
    )

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_orientation = "vertical"
    p.add_layout(color_bar, 'right')
    p.add_tools(HoverTool(tooltips=[(grid.index.name, f"@{grid.index.name}"), (grid.columns.name, f"@{grid.columns.name}")
        , ("value", "@value")]))

    return p