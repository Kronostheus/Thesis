import pickle
import plotly.graph_objects as go
import plotly.express as px
from Utils.config import Config
from plotly.subplots import make_subplots
from string import punctuation
from operator import itemgetter
from collections import Counter

# for now, only 20 colors. Increase if more categories needed
colors = px.colors.qualitative.Vivid[:-1] + px.colors.qualitative.Plotly
color_dict = {'Others': px.colors.qualitative.Vivid[-1]}

# correct, incorrect = pickle.load(open(Config.DATA_DIR + 'explanations_test.pickle', 'rb'))
articles = pickle.load(open(Config.DATA_DIR + 'explanations_articles.pickle', 'rb'))
right_margin = 0.0  # if needed

for text, top_labels, lime in articles:
# for text, top_labels, lime in correct + incorrect:
    topics, freqs = zip(*top_labels)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{'colspan': 2, 'r': right_margin}, None],
            [{}, {}]
        ],
        subplot_titles=[
            # 'Most Probable Categories',
            # 'Word impact for: {}'.format(topics[0]),
            # 'Word impact for: {}'.format(topics[1])
            'Categorias Mais Prováveis',
            'Impacto para: {}'.format(topics[0]),
            'Impacto para: {}'.format(topics[1]),
        ],
        row_width=[5, 1],
        vertical_spacing=0.2,
        horizontal_spacing=0.01,
        shared_yaxes=True
    )

    if freqs[-1] < 0.1:
        freqs = freqs[:-1] + (1 - sum(freqs[:-1]),)
        topics = topics[:-1] + ('Outros',)
    else:
        freqs += (1 - sum(freqs),)
        topics += ('Outros',)

    for topic in topics:
        if topic not in color_dict.keys():
            color_dict[topic] = colors.pop()

    fig.add_trace(
        go.Bar(
            x=freqs,
            y=['' for f in freqs],
            orientation='h',
            width=2,
            text=['{}<br>{}%'.format(topic, round(freqs[i] * 100, 2)) for i, topic in enumerate(topics)],
            textposition='inside',
            insidetextanchor='middle',
            # marker_color=[color_dict[topic] for topic in topics],
            marker=dict(
                color=[color_dict[topic] for topic in topics],
            ),
            textfont=dict(
                size=17
            ),
            showlegend=False
        ),
        row=1,
        col=1
    )

    text_words = [word for word in text.translate(str.maketrans('', '', punctuation)).split() if word]
    text_lst = list(dict.fromkeys(text_words))
    cnt = Counter(text_words)
    for idx, (label, weights) in enumerate(lime):

        sorted_weights = [weight for i, weight in sorted(weights, key=itemgetter(0))]

        for word, word_cnt in cnt.items():
            if word_cnt > 1:

                # Get all indices of this word
                idxs = [idx for idx, w in enumerate(text_words) if w == word]

                # Get index of word that is present in weight list
                org_idx = text_lst.index(word)

                # Divide equally the weight over all indices of the word
                new_weight = sorted_weights[org_idx] / len(idxs)

                # Set old weight to new word
                sorted_weights[org_idx] = new_weight

                # Iterate through remaining indices of the word and insert them into weight list
                for word_index in idxs[1:]:
                    sorted_weights.insert(word_index, new_weight)

        assert len(text_words) == len(sorted_weights)

        fig.add_trace(
            go.Bar(
                x=[str(i) for i, _ in enumerate(text_words)],
                y=sorted_weights,
                marker=dict(
                    color=['crimson' if weight < 0 else 'olivedrab' for weight in sorted_weights]
                ),
                showlegend=False
            ),
            row=2,
            col=idx+1
        )

    fig.update_xaxes(
                    row=1,
                    col=1,
                    visible=False,
                    range=[0, 1]
                    )

    fig.update_xaxes(
                    row=2,
                    ticktext=text_words,
                    tickvals=[i for i, _ in enumerate(text_words)],
                    tickfont=dict(size=18),
                    tickangle=-30
                    )

    fig.update_layout()

    fig.update_layout(
        # title="Explanation for '{}'".format(text),
        title=dict(
            text="Explanation for '{}'".format(text),
            font=dict(size=22)
        ),
        width=1980,
        height=600,
        legend=dict(
                orientation='h',
                yanchor="bottom",
                y=-0.1,
                xanchor="right",
                x=0.85
        ),
    )

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20)

    fig.show()
