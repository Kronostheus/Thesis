import pandas as pd
import numpy as np
import torch
import random
from simpletransformers.classification import ClassificationModel
from simpletransformers.classification.transformer_models.bert_model import BertForSequenceClassification
from simpletransformers.classification.classification_utils import (
    InputExample,
    convert_examples_to_features,
)
from simpletransformers.config.global_args import global_args
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import os
import math
import json
import warnings
import logging

from multiprocessing import cpu_count

from scipy.stats import pearsonr, mode
from sklearn.metrics import (
    mean_squared_error,
    matthews_corrcoef,
    confusion_matrix,
    label_ranking_average_precision_score,
)
from tensorboardX import SummaryWriter
from tqdm.auto import trange, tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

# Used to find out which GPU is being used in Google Colaboratory (training done when GPU has 17GB of total_memory)
t = torch.cuda.get_device_properties(0).total_memory
print('Total Memory: {}'.format(t / 1e9))

"""
DATA PREPARATION
"""

# There are a total of 46 categories which need to be converted
le = LabelEncoder()

train = pd.read_csv('train.csv', names=["text", "Code"], header=0)
train["labels"] = le.fit_transform(train.Code)
train_df = train[["text", "labels"]]

val = pd.read_csv('val.csv', names=["text", "Code"], header=0)
val["labels"] = le.fit_transform(val.Code)
val_df = val[["text", "labels"]]

test = pd.read_csv('test.csv', names=["text", "Code"], header=0)
test["labels"] = le.fit_transform(test.Code)
test_df = test[["text", "labels"]]


class MetaBertModel(ClassificationModel):
    """
    This is nothing more than the original ClassificationModel from simpletransformers which is based on the run_glue.py
    example from HuggingFace's Transformers library, on which it is built.
    Only difference is the inclusion of two lists that hold the hidden states from the last BERT layer of the last
    epoch. These are to be used in several experiments involving meta-learning as seen later on in this file.
    """
    def __init__(
        self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):
        MODEL_CLASSES = {
                  "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
              }

        if args and 'manual_seed' in args:
            random.seed(args['manual_seed'])
            np.random.seed(args['manual_seed'])
            torch.manual_seed(args['manual_seed'])
            if 'n_gpu' in args and args['n_gpu'] > 0:
                torch.cuda.manual_seed_all(args['manual_seed'])

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(model_name,
                                                       num_labels=num_labels,
                                                       output_hidden_states=True,
                                                       **kwargs)
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name,
                                                       output_hidden_states=True,
                                                       **kwargs)
            self.num_labels = self.config.num_labels

        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:

            self.model = model_class.from_pretrained(
                model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs,
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)

        self.results = {}

        self.args = {
            "sliding_window": False,
            "tie_value": 1,
            "stride": 0.8,
            "regression": False,
        }

        self.args.update(global_args)

        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args["do_lower_case"], **kwargs)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None
        
        self.hidden_states = []
        self.eval_states = []

    def process_batch(self, batch, labels):

        new_matrix = [np.matrix(batch_element).max(0).A1 for batch_element in batch]

        return tuple(zip(new_matrix, labels))
    
    def train(
        self,
        train_dataset,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total
        )

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"])
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0

        if args["evaluate_during_training"]:
            training_progress_scores = self._create_training_progress_scores(multi_label, **kwargs)

        if args["wandb_project"]:
            wandb.init(project=args["wandb_project"], config={**args}, **args["wandb_kwargs"])
            wandb.watch(self.model)

        model.train()
        for _ in train_iterator:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args["silent"])):
                batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                outputs = model(**inputs)

                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                if epoch_number == args["num_train_epochs"] - 1:
                  self.hidden_states.extend(
                      self.process_batch(outputs[2][-1].detach().cpu().numpy(), inputs["labels"].detach().cpu().numpy())
                  )

                if args["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    if args["fp16"]:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args["logging_steps"], global_step)
                        logging_loss = tr_loss
                        if args["wandb_project"]:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self._save_model(output_dir_current, model=model)

                    if args["evaluate_during_training"] and (
                        args["evaluate_during_training_steps"] > 0
                        and global_step % args["evaluate_during_training_steps"] == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_df, save_hidden=False, verbose=verbose and args["evaluate_during_training_verbose"], silent=True, **kwargs
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args["save_eval_checkpoints"]:
                            self._save_model(output_dir_current, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False,
                        )

                        if args["wandb_project"]:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args["early_stopping_metric"]]
                            self._save_model(args["best_model_dir"], model=model, results=results)
                        if best_eval_metric and args["early_stopping_metric_minimize"]:
                            if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(args["best_model_dir"], model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args['early_stopping_metric']}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args['early_stopping_patience']} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                        else:
                            if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(args["best_model_dir"], model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args['early_stopping_metric']}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args['early_stopping_patience']} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args["save_model_every_epoch"] or args["evaluate_during_training"]:
                os.makedirs(output_dir_current, exist_ok=True)

            if args["save_model_every_epoch"]:
                self._save_model(output_dir_current, model=model)

            if args["evaluate_during_training"]:
                results, _, _ = self.eval_model(
                    eval_df, save_hidden=False, verbose=verbose and args["evaluate_during_training_verbose"], silent=False, **kwargs
                )

                self._save_model(output_dir_current, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args["output_dir"], "training_progress_scores.csv"), index=False)

                if not best_eval_metric:
                    best_eval_metric = results[args["early_stopping_metric"]]
                    self._save_model(args["best_model_dir"], model=model, results=results)
                if best_eval_metric and args["early_stopping_metric_minimize"]:
                    if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], model=model, results=results)
                        early_stopping_counter = 0
                else:
                    if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], model=model, results=results)
                        early_stopping_counter = 0

        return global_step, tr_loss / global_step

    def evaluate(self, eval_df, output_dir, save_hidden=False, multi_label=False, prefix="", verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_df.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}

        if "text" in eval_df.columns and "labels" in eval_df.columns:
            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(zip(eval_df["text"], eval_df["labels"]))
            ]
        elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
            eval_examples = [
                InputExample(i, text_a, text_b, label)
                for i, (text_a, text_b, label) in enumerate(
                    zip(eval_df["text_a"], eval_df["text_b"], eval_df["labels"])
                )
            ]
        else:
            warnings.warn(
                "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
            )
            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))
            ]

        if args["sliding_window"]:
            eval_dataset, window_counts = self.load_and_cache_examples(
                eval_examples, evaluate=True, verbose=verbose, silent=silent
            )
        else:
            eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True, verbose=verbose, silent=silent)
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"] or silent):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if save_hidden:
                  self.eval_states.extend(
                        self.process_batch(outputs[2][-1].detach().cpu().numpy(), inputs["labels"].detach().cpu().numpy())
                  )

                if multi_label:
                    logits = logits.sigmoid()
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args["sliding_window"]:
            count = 0
            window_ranges = []
            for n_windows in window_counts:
                window_ranges.append([count, count + n_windows])
                count += n_windows

            preds = [preds[window_range[0] : window_range[1]] for window_range in window_ranges]
            out_label_ids = [
                out_label_ids[i] for i in range(len(out_label_ids)) if i in [window[0] for window in window_ranges]
            ]

            model_outputs = preds

            preds = [np.argmax(pred, axis=1) for pred in preds]
            final_preds = []
            for pred_row in preds:
                mode_pred, counts = mode(pred_row)
                if len(counts) > 1 and counts[0] == counts[1]:
                    final_preds.append(args["tie_value"])
                else:
                    final_preds.append(mode_pred[0])
            preds = np.array(final_preds)
        elif not multi_label and args["regression"] is True:
            preds = np.squeeze(preds)
            model_outputs = preds
        else:
            model_outputs = preds

            if not multi_label:
                preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(preds, out_label_ids, eval_examples, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results, model_outputs, wrong

    def eval_model(self, eval_df, save_hidden=False, multi_label=False, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_df. Saves results to output_dir.
        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        result, model_outputs, wrong_preds = self.evaluate(
            eval_df, output_dir, save_hidden=save_hidden, multi_label=multi_label, verbose=verbose, silent=silent, **kwargs
        )
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, wrong_preds


# Model and respective args to use
model = MetaBertModel('bert',
                      'bert-base-multilingual-cased',
                      num_labels=len(train_df.labels.unique()),
                      args={'train_batch_size': 32,
                            'eval_batch_size': 32,
                            'num_train_epochs': 4,
                            'max_seq_length': 200,
                            'save_steps': 0,
                            'evaluate_during_training': True,
                            'evaluate_during_training_steps': int(train_df.shape[0] / 32),
                            'evaluate_during_training_verbose': True,
                            'fp16': False,
                            'overwrite_output_dir': True,
                            'reprocess_input_data': True,
                            'learning_rate': 2e-5,
                            'manual_seed': 42
                            })

# When GPU memory = 17GB this takes around 45-55 minutes per epoch
model.train_model(train_df=train_df, eval_df=val_df)


def f_macro(true, preds):
    """
    Macro averaged F-score
    :param true: true labels
    :param preds: predicted labels
    :return: Macro-averaged F-score
    """
    return metrics.f1_score(true, preds, average='macro')


def f_micro(true, preds):
    """
    Micro averaged F-score (in our case, this is equivalent to the accuracy)
    :param true: true labels
    :param preds: predicted labels
    :return: Micro-averaged F-score
    """
    return metrics.f1_score(true, preds, average='micro')


def softmax(logits):
    """
    Softmax algorithm in order to normalize logits
    :param logits: list of logits
    :return: list of probabilities
    """
    return np.exp(logits) / np.sum(np.exp(logits), axis=0)


# Should take about 5-10 minutes
result, model_outputs, wrong_predictions = model.eval_model(test_df,
                                                            save_hidden=True,
                                                            acc=metrics.accuracy_score,
                                                            f1M=f_macro,
                                                            f1m=f_micro)

print(result)

train_vectores = model.hidden_states
test_vectores = model.eval_states

train_vecs, train_lbs = zip(*train_vectores)

test_vecs, test_lbs = zip(*test_vectores)

# Generate CSV files so I don't have to run a model everytime. Training CSV will exceed 2GB, though.
"""train_df = pd.DataFrame.from_records(train_vecs)
train_df["label"] = train_lbs
train_df.head()

test_df = pd.DataFrame.from_records(test_vecs)
test_df["label"] = test_lbs
test_df.head()

train_df.to_csv("hidden_train.csv", index=False)
test_df.to_csv("hidden_test.csv", index=False)"""

def print_metrics(true, preds):
    """
    Group printing of metrics into a simpler method.
    :param true: true labels
    :param preds: predictions
    :return:
    """
    print("ACC: {}\nF-Score: {}\nMCC: {}".format(
      metrics.accuracy_score(true, preds),
      metrics.f1_score(true, preds, average='macro'),
      metrics.matthews_corrcoef(true, preds)
    ))

# Group categories into respective domains
domain_to_label = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 9: []}
label_to_code = dict(zip(train.labels, train.Code))
for label, code in label_to_code.items():
    domain = code // 100
    domain_to_label[domain].append(label)


def get_domains(labels_lst):
    result = []

    for label in labels_lst:
        for domain, labels in domain_to_label.items():
            if label in labels:
                result.append(domain)

    return result


"""
META-LEARNING
"""

knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', n_jobs=-1)
knn.fit(train_vecs, train_lbs)

preds = knn.predict(test_vecs)

print("KNN categories")
print_metrics(test_lbs, preds)

preds_domains = get_domains(preds)
true_domains = get_domains(test.labels)

print("KNN Domains")
print_metrics(true_domains, preds_domains)

preds_bert = [np.argmax(softmax(logits)) for logits in model_outputs]
preds_domains_bert = get_domains(preds_bert)

print("BERT Domains")
print_metrics(true_domains, preds_domains_bert)

lr = LogisticRegression(penalty='l2', random_state=42, n_jobs=-1)
lr.fit(train_vecs, train_lbs)

lr_preds = lr.predict(test_vecs)

print("LR categories")
print_metrics(test_lbs, lr_preds)

preds_domains_lr = get_domains(lr_preds)

print("LR domains")
print_metrics(true_domains, preds_domains_lr)

lr_ = LogisticRegression(penalty='none', random_state=42, n_jobs=-1)
lr_.fit(train_vecs, train_lbs)

lr__preds = lr_.predict(test_vecs)


print("LR - no penalty")
print_metrics(test_lbs, lr__preds)

preds_domains_lr2 = get_domains(lr__preds)

print("LR domains - no penalty")
print_metrics(true_domains, preds_domains_lr2)
