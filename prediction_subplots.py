import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from Utils.config import Config
from plotly.subplots import make_subplots


class Visual:
    def __init__(self, data, man_labels):

        df = data[~data.code.isin([0, 999])]

        self.verbose_preds = df.astype(str).merge(man_labels.astype(str), on='code')
        self.verbose_preds['domain'] = self.verbose_preds.code.apply(lambda c: str(c)[0])

        self.man_labels = man_labels

        self.domain_verbose = {
            1: 'External Relations',
            2: 'Freedom and Democracy',
            3: 'Political System',
            4: 'Economy',
            5: 'Welfare and Quality of Life',
            6: 'Fabric of Society',
            7: 'Social Groups',
            # 9: 'Non-Relevant Span'
        }

    def get_domains(self, topics):
        return [str(self.man_labels[self.man_labels.Topic == topic].code.values[0])[0] for topic in topics]

    def get_codes(self, topics):
        return [int(self.verbose_preds[self.verbose_preds.Topic == t].topic.unique()[0]) for t in topics]

    def get_colors(self, topics):
        return [px.colors.qualitative.Plotly[dom] for dom in self.get_domains(topics)]

    def draw(self, legend=False):

        freqs = self.verbose_preds.Topic.value_counts(normalize=True, sort=False)
        y, x = list(freqs.index), [round(freq, 4) for freq in freqs.values]

        for topic in self.man_labels.Topic.unique():
            if topic not in y and topic not in ('General', 'Header'):
                y.append(topic)
                x.append(0.0)

        domain_indices = {idx: [i for i, dom in enumerate(self.get_domains(y)) if dom == str(idx)]
                          for idx, domain in list(self.domain_verbose.items())[::-1]}

        traces = []

        for domain, indices in domain_indices.items():
            x_d = [x[i] for i in indices]
            y_d = [y[i] for i in indices]

            traces.append(go.Bar(
                    x=x_d,
                    y=y_d,
                    marker_color=px.colors.qualitative.Vivid[domain],
                    name=self.domain_verbose[domain],
                    orientation='h',
                    showlegend=legend
                ))

        return traces


man = pd.read_csv(Config.SCHEMES_DIR + 'final_man.csv', names=['Topic', 'code'], header=0)

authors = os.listdir(Config.PRED_DIR)

fig = make_subplots(rows=1, cols=5, shared_yaxes=True, subplot_titles=authors, horizontal_spacing=0.02, print_grid=False)

for idx, author in enumerate(authors):

    data = pd.read_csv(Config.PRED_DIR + author + '/results.csv')
    trace_lst = Visual(data, man_labels=man).draw(idx == 4)

    for trace in trace_lst:
        fig.add_trace(trace, row=1, col=idx + 1)

    fig.update_xaxes(tickformat='%', dtick=0.05, range=[0, 0.35], row=1, col=idx + 1)

fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor="bottom",
        y=-0.1,
        xanchor="right",
        x=0.85
    )
)

fig.show()