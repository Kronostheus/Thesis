import pandas
from Eval.ner_evaluation.ner_eval import Entity, Evaluator
from Utils import get_spans

df = pandas.read_csv('pipeline_results.csv')
# df = df[df.country == 'S']


def f_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0


# df["true_domain"] = df.true_code.apply(lambda x: str(x)[0])
# df["pred_domain"] = df.pred_code.apply(lambda x: str(x)[0])

true = []
pred = []

t = 0
p = 0

for _, sent_df in df.groupby(by='sid'):

    true_spans = get_spans(sent_df.true_id.to_list(), sent_df.true_code.to_list())
    pred_spans = get_spans(sent_df.pred_id.to_list(), sent_df.pred_code.to_list())
    # true_spans = get_spans(sent_df.true_id.to_list(), sent_df.true_domain.to_list())
    # pred_spans = get_spans(sent_df.pred_id.to_list(), sent_df.pred_domain.to_list())

    true_ents = []
    for start, end in zip(true_spans, true_spans[1:]):
        t += 1
        span_type = str(sent_df[start:end].true_code.value_counts().keys().to_list()[0])
        # span_type = str(sent_df[start:end].true_domain.value_counts().keys().to_list()[0])
        # span_type = "span" if sent_df[start:end].true_id.to_list()[0] == 'B' else "outside"
        true_ents.append(Entity(span_type, start, end))

    pred_ents = []
    for start, end in zip(pred_spans, pred_spans[1:]):
        p += 1
        span_type = str(sent_df[start:end].pred_code.value_counts().keys().to_list()[0])
        # span_type = str(sent_df[start:end].pred_domain.value_counts().keys().to_list()[0])
        # span_type = "span" if sent_df[start:end].pred_id.to_list()[0] == 'B' else "outside"
        pred_ents.append(Entity(span_type, start, end))

    true.append(true_ents)
    pred.append(pred_ents)

tags = [str(c) for c in df.true_code.unique()]
# tags = [str(c) for c in df.true_domain.unique()]

evaluator = Evaluator(true, pred, tags)
# evaluator = Evaluator(true, pred, ["span", "outside"])
results, results_agg = evaluator.evaluate()

# f_partial = f_score(results['partial']['precision'], results['partial']['recall'])
# f_exact = f_score(results['exact']['precision'], results['exact']['recall'])
f_scores = {res: f_score(results[res]['precision'], results[res]['recall']) for res in results.keys()}
breakpoint()


