import math


class record:
    def __init__(self, origin, pos, neg, right, predict, similarity, importance):
        self.origin = origin
        self.pos = pos
        self.neg = neg
        self.right = right
        self.predict = predict
        self.importance = importance
        self.similarity = similarity

    def __repr__(self):
        return "origin={},pos={},neg={},right={},result={}\n".format(self.origin, self.pos, self.neg, self.right,
                                                                     self.predict)


class result:
    def __init__(self, recall, ndcg, acc, similarity, new):
        self.recall = recall
        self.ndcg = ndcg
        self.acc = acc
        self.similarity = similarity
        self.new = new

    def __repr__(self):
        return "recall={},ndgc={},acc={},similarity={},new={}\n".format(self.recall, self.ndcg, self.acc,
                                                                        self.similarity, self.new)


def find_array(array):
    temp = 0
    ans = []
    for arr in array:
        for x in arr:
            if '0' <= x <= "9":
                ans.append(int(x))
    return ans


def find_num(s):
    ans = 0
    for x in s:
        if '0' <= x <= "9":
            ans = int(x) + ans * 10
    return ans


def get_record(file_path):
    records = []
    with open(file_path) as f:
        while True:
            line = f.readline()
            if line:
                s = line.split(",")
                try:
                    origin = find_num(s[3])
                    pos = find_num(s[4])
                    neg = find_num(s[5])
                    right = find_num(s[6])
                    predict = find_array(s[7:-2])
                    similarity = float(s[-2][11:])
                    new = float(s[-1][4:])
                    records.append(record(origin, pos, neg, right, predict, similarity, new))
                except IndexError:
                    break
            else:
                break
    return records


def NDCG(file_path):
    records = get_record(file_path)
    result = 0
    cnt = 0
    for val in records:
        temp = 0
        temp2 = 0
        pos = 0
        for idx, x in enumerate(val.predict):
            pos += x
            temp += x / math.log(idx + 2)
        for x in range(20):
            temp2 += 1 / math.log(x + 2)
        if temp2 != 0:
            result += temp / temp2
            cnt += 1
    return result / cnt


def similarity(file_path):
    records = get_record(file_path)
    result = 0
    cnt = 0
    for val in records:
        result += val.similarity
        cnt += 1
    return result / cnt


def new(file_path):
    records = get_record(file_path)
    result = 0
    cnt = 0
    for val in records:
        result += val.importance
        cnt += 1
    return result / cnt


def recall(file_path):
    records = get_record(file_path)
    result = 0
    cnt = 0
    for val in records:
        # if val.pos == 0: continue
        result += val.right / val.origin
        cnt += 1
    return result / cnt


def acc(file_path):
    records = get_record(file_path)
    result = 0
    cnt = 0
    for val in records:
        result += val.right / len(val.predict)
        cnt += 1
    return result / cnt


def get_evaluate(file_path):
    return result(recall(file_path), NDCG(file_path), acc(file_path),similarity(file_path),new(file_path))


if __name__ == '__main__':
    file_path = "evaluate/movielen/GIN/batch_10sample_50/recommend_80origin_20/alpha_0.5bata_1.0hop_2max_5"
    print(get_evaluate(file_path))
