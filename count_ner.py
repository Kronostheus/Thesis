from Utils import Config, get_spans, get_spans_old
import pandas as pd

train = pd.read_csv(Config.TRAIN_DIR + 'train_ner.csv')
val = pd.read_csv(Config.VAL_DIR + 'val_ner.csv')
test = pd.read_csv(Config.TEST_DIR + 'test_ner.csv')
# aug = pd.read_csv(Config.TRAIN_DIR + 'train_ner_translated.csv')

for _, sent_df in train.groupby(by='sentence_id'):
    if 0 in list(sent_df.codes.unique()) and sent_df.country.unique()[0] == 'P':
        breakpoint()

country = ''

if country:
    train = train[train.country == country]
    test = test[test.country == country]
    val = val[val.country == country]

sentences_train = len(train.sentence_id.unique())
sentences_test = len(test.sentence_id.unique())
sentences_val = len(val.sentence_id.unique())
# sentences_aug = len(aug.sentence_id.unique())

# spans = {'train': 0, 'test': 0, 'val': 0, 'aug': 0}
# multi = {'train': 0, 'test': 0, 'val': 0, 'aug': 0}
spans = {'train': 0, 'test': 0, 'val': 0}
multi = {'train': 0, 'test': 0, 'val': 0}
single = {'train': 0, 'test': 0, 'val': 0}

# tags = {'train': [], 'test': [], 'val': [], 'aug': []}
tags = {'train': [], 'test': [], 'val': []}


# for df, df_name in zip((train, test, val, aug), ('train', 'test', 'val', 'aug')):
for df, df_name in zip((train, test, val), ('train', 'test', 'val')):
    for _, sent_df in df.groupby(by='sentence_id'):
        tags[df_name].append(len(sent_df))

        # boundaries = [(start, end) for start, end in get_spans(sent_df.labels)]

        t = get_spans_old(sent_df.labels, sent_df.codes)
        boundaries = [(start, end) for start, end in zip(t, t[1:])]

        spans[df_name] += len(boundaries)
        multi[df_name] += len(boundaries) if len(boundaries) > 1 else 0
        single[df_name] += 1 if len(boundaries) == 1 else 0


tags = {df_name: sum(lens) / len(lens) for df_name, lens in tags.items()}

breakpoint()