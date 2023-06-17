import os.path

from occurrence_graph import *


class record:
    def __init__(self, intention, infomax):
        self.intention = intention
        self.infomax = infomax

    def __lt__(self, other):
        return self.infomax < other.infomax


class parameter:
    def __init__(self, dataset, alpha, beta, hop, model, batch, sample, maxx):
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
        self.hop = hop
        self.model = model
        self.batch = batch
        self.sample = sample
        self.max = maxx

    def __repr__(self):
        return "alpha={},beta={},dataset={},hop={},max={}".format(self.alpha, self.beta, self.dataset, self.hop,
                                                                  self.max)


def get_path(path):
    args = path.split("/")
    dataset = args[1]
    model = args[2]
    jack = args[3].split("_")
    alpha = float(jack[0])
    beta = float(jack[1])
    hop = int(jack[2])
    jack = args[4].split("_")
    sample = int(jack[5])
    batch = int(jack[3])
    maxx = int(jack[7])
    return parameter(dataset, alpha, beta, hop, model, batch, sample, maxx)


def recall(path, epoch=100, top=20, device='cuda', base=20, recommend=80):
    args = get_path(path)
    intention_size = 100
    user_size = 50
    if args.dataset == "yelp":
        user_size = 30
        intention_size = 100
    model = torch.load(path, map_location=device).to(device)
    array = list()
    model.to(device)
    model.eval()
    sum_recall = 0
    max_recall = 0
    max_acc = 0
    sum_acc = 0
    with torch.no_grad():
        for k in range(epoch):
            print(path)
            true_cnt = 0
            origin = 0
            vis = dict()
            vis.clear()
            user = get_user(False, max_intention=user_size, min_intention=user_size, dataset=args.dataset,
                            hop=args.hop, beta=args.beta, max_adaption=args.max)
            user.feature = model.encoder(user.origin_PYG_data.to(device), device)
            # print("user intention:{}".format(len(user.intention_list)))
            # print("user item list:{}".format(len(user.item_list)))
            intention_list = []
            for intention in user.intention_list:
                if origin > base: break
                intention_list.append(intention)
                for item in intention.node_list:
                    if vis.get(item.id) is None:
                        vis[item.id] = 1
                        origin += 1
            vis.clear()
            temp_list = get_random_intention(intention_size=intention_size, dataset=args.dataset, hop=args.hop,
                                             beta=args.beta, max_adaption=args.max)
            pos = 0
            neg = 0
            predict = list()
            for intention in temp_list:
                if pos + neg > recommend: break
                intention_list.append(intention)
                for item in intention.node_list:
                    if vis.get(item.id) is not None: continue
                    vis[item.id] = 1
                    if item.id in user.item_list:
                        pos += 1
                    else:
                        neg += 1

            for intention in intention_list:
                intention.aggr_feature = model.encoder(intention.PYG_data().to(device), device, True)
                predict.append(record(intention, model.get_infomax(user, intention)))
            predict = sorted(predict)
            vis.clear()
            cnt = 0
            result = []
            items = []
            new = []
            similarity = 0
            for val in predict:
                # input(val.infomax)
                if cnt == top: break
                for item in val.intention.node_list:
                    if vis.get(item.id) is not None: continue
                    vis[item.id] = 1
                    cnt += 1
                    if item.id in user.item_list:
                        true_cnt += 1
                        result.append(1)
                    else:
                        result.append(0)
                    items.append(item.feature)
                    new.append(item.importance)
                    if cnt == top: break
            for x in items:
                for y in items:
                    similarity += F.cosine_similarity(x, y, dim=1)
            recall = true_cnt / origin
            acc = true_cnt / top
            max_acc = max(max_acc, acc)
            sum_acc += acc
            max_recall = max(max_recall, recall)
            sum_recall += recall
            array.append(
                "batch{}:recall={},acc={},k = {},origin={},pos={},neg={},right = {},result={},similarity={},new={}\n"
                .format(k, recall, acc, top, origin, pos, neg, true_cnt, result, float(similarity), sum(new)))
            print(
                "batch{}:recall={},acc={},k = {},origin={},pos={},neg={},right = {},result={},similarity={},new={}"
                .format(k, recall, acc, top, origin, pos, neg, true_cnt, result, float(similarity), sum(new)))
    array.append("avg_recall={},max_recall={},avg_acc={},max_acc={},pos_intention={},neg_intention={}\n".format(
        (sum_recall / epoch), max_recall, (sum_acc / epoch), max_acc, user_size,
        intention_size))
    print("avg_recall={},max_recall={},avg_acc={},max_acc={},pos_intention={},neg_intention={}".format(
        (sum_recall / epoch), max_recall, (sum_acc / epoch), max_acc, user_size,
        intention_size))
    name = "alpha_{}bata_{}hop_{}max_{}".format(args.alpha, args.beta, args.hop, args.max)
    path = "evaluate/{}/{}/batch_{}sample_{}/recommend_{}origin_{}".format(args.dataset, args.model, args.batch,
                                                                           args.sample, recommend, base)
    os.makedirs(path, exist_ok=True)
    f = open(os.path.join(path, name), mode="w")
    for i in array:
        f.write(i)
    f.close()


if __name__ == '__main__':
    path = "save/movielen/GIN/0.3_1_2/epoch_50_batch_10_sample_50_max_4"
    recall(path, device="cuda:2")
    # recall(path, device="cuda", dataset="movielen")
