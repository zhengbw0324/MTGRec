import json
import os
import argparse
from collections import defaultdict

import torch




class Evaluator:
    def __init__(self, metrics, topk):

        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }
        self.metrics = metrics
        self.topk = topk

        self.maxk = max(topk)

    def ensemble_results(self, results_info_list):
        labels = results_info_list[0]['label_ids']
        N = len(labels)
        preds = [defaultdict(list) for _ in range(N)]

        for results_info in results_info_list:
            # print(results_info['pred_ids'][0])
            # print(results_info['scores'][0])
            # print("===========================")
            pred_ids = results_info['pred_ids']
            scores = results_info['scores']
            label_ids = results_info['label_ids']
            for i in range(N):
                item_list = pred_ids[i]
                score_list = scores[i]
                assert label_ids[i] == labels[i]
                for item, score in zip(item_list, score_list):
                    preds[i][item].append(score)

        # print(preds[0])
        # # preds = [sorted(pred.items(), key=lambda x: x[1], reverse=True) for pred in preds]
        # # # preds = [item  for pred in preds for item, _ in pred]
        # # preds = [[item for item, _ in pred] for pred in preds]
        # final_preds = []
        # for pred in preds:
        #     sorted_items = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        #     final_preds.append([item for item, _ in sorted_items[:self.maxk]])

        final_preds = []
        for i in range(N):
            pred = defaultdict(float)
            for item, score_list in preds[i].items():
                pred[item] = sum(score_list) / len(score_list)
                if item!= "None":
                    pred[item] += len(score_list)

            sorted_items = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            # if i==0:
            #     print(sorted_items)
            pred = [item for item, _ in sorted_items]
            if len(pred) < self.maxk:
                pred += ["None"] * (self.maxk - len(pred))
            final_preds.append(pred)




        return final_preds, labels


    def calculate_pos_index(self, preds, labels):

        N = len(preds)
        # print(N)
        # print(preds[0])
        pos_index = torch.zeros((N, self.maxk), dtype=torch.bool)
        for i in range(N):
            cur_label = labels[i]
            for j in range(self.maxk):
                try:
                    cur_pred = preds[i][j]
                except Exception as e:
                    print(e)
                    print(i)
                    print(preds[i])
                    print(j)
                    raise RuntimeError
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.metrics:
            for k in self.topk:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        results = {k: v.mean().item() for k, v in results.items()}
        return results


def main(args):
    metrics = args.metrics
    topk = args.topk

    evaluator = Evaluator(metrics, topk)
    # temp = []
    results_file_list = os.listdir(args.results_dir)
    # results_file_list = [
    #     "test_results_9987.json",
    #     "test_results_9988.json",
    #     "test_results_9989.json",
    #     "test_results_9990.json",
    #     "test_results_9995.json",
    # ]
    results_file_list = [f for f in results_file_list if f.endswith('.json')]
    results_file_list.sort()
    results_info_list = []
    max_results = {}
    for results_file in results_file_list:
        results_info = json.load(open(os.path.join(args.results_dir, results_file), 'r'))
        results_info_list.append(results_info)
        temp_res = evaluator.calculate_metrics(results_info['pred_ids'], results_info['label_ids'])
        print(f"Results for {results_file}:")
        print(temp_res)
        for k, v in temp_res.items():
            if k not in max_results:
                max_results[k] = v
            else:
                max_results[k] = max(max_results[k], v)

    print("Max Results:")
    print(max_results)
    preds, labels = evaluator.ensemble_results(results_info_list)
    metrics_results = evaluator.calculate_metrics(preds, labels)
    print("Ensemble Results:")
    print(metrics_results)





def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument("--results_dir", type=str,
    default="./results/Video_Games/"
    )

    parser.add_argument("--metrics", type=str, default="ndcg,recall")
    parser.add_argument("--topk", type=str, default="5,10")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.metrics = args.metrics.split(",")
    args.topk = list(map(int, args.topk.split(",")))
    main(args)



