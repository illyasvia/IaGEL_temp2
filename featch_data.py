import pandas as pd
import torch


class link:
    def __init__(self, user, item, timestamp, value=None):
        self.user = user
        self.item = item
        self.value = value
        self.time = timestamp

    def __repr__(self):
        return "{}->{}:value = {},time = {}\n".format(self.user, self.item, self.value, self.time)

    def __lt__(self, other):
        return self.time < other.time


def string_to_tensor(string):
    array = []
    temp = ''
    for x in string:
        if '0' <= x <= '9':
            temp += x
        else:
            if temp != '':
                array.append(int(temp))
                temp = ''
    array = torch.tensor(array, dtype=torch.float32)
    array = torch.unsqueeze(array, 0)
    return array


def string_to_tensor_yelp(string):
    array = []
    temp = ''
    for x in string:
        if '0' <= x <= '9':
            temp += x
        else:
            if temp != '':
                array.append(int(temp))
                temp = ''
    if not array: return None
    tensor = torch.zeros(array[-2] + 1, dtype=torch.float32)
    tensor[array[-2]] = array[-1]
    for x in range(0, len(array) - 2):
        tensor[array[x]] = 1
    return torch.unsqueeze(tensor, 0)


def get_from_csv(dataset='movielen'):
    flag = (dataset == 'movielen')
    if dataset == 'movielen':
        csv = pd.read_csv("../data/movielen/item.csv")
    elif dataset == 'yelp':
        csv = pd.read_csv("../data/yelp/item.csv")
    elif dataset == 'amazon':
        csv = pd.read_csv("../data/amazon/item.csv")
    nodes = dict()
    for index in csv.index:
        temp = csv.loc[index]
        if flag:
            feature = string_to_tensor(temp['feature'])
        else:
            feature = string_to_tensor_yelp(temp['feature'])
        if feature is None: continue
        nodes[temp['id']] = feature

    if dataset == 'movielen':
        csv = pd.read_csv("../data/movielen/user.csv")
    elif dataset == 'yelp':
        csv = pd.read_csv("../data/yelp/user.csv")
    elif dataset == 'amazon':
        csv = pd.read_csv("../data/amazon/user.csv")
    users = dict()
    for index in csv.index:
        temp = csv.loc[index]
        feature = string_to_tensor(temp['feature'])
        users[temp['id']] = feature
        if not flag:
            if len(users.keys()) >= 10000 - 1:
                break

    if dataset == 'movielen':
        csv = pd.read_csv("../data/movielen/edge.csv")
    elif dataset == 'yelp':
        csv = pd.read_csv("../data/yelp/edge.csv")
    elif dataset == 'amazon':
        csv = pd.read_csv("../data/amazon/edge.csv")
    links = list()
    for index in csv.index:
        temp = csv.loc[index]
        user_id = temp['user']
        item_id = temp['item']
        time = temp['time']
        if nodes.get(item_id) is None or users.get(user_id) is None: continue
        links.append(link(user_id, item_id, time))
    # print("user:{},nodes:{},links:{}".format(len(users.keys()), len(nodes.keys()), len(links)))
    return nodes, users, links


if __name__ == '__main__':
    get_from_csv()
