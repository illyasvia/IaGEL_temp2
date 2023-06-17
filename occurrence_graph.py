import random
from copy import copy
from random import choice

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from featch_data import get_from_csv
from train import *


class intention:
    def __init__(self, node_list, edge_index, edge_attr, edge_weight, hop=1):
        self.node_list = node_list
        self.aggr_feature = None
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_weight = edge_weight
        self.hop = hop
        self.importance = 0
        self.similarity = 0
        self.update()
        self.feature = None

    def modularity(self, graph: nx.Graph):
        m = 0
        w = {}
        result = 0
        for node1 in self.node_list:
            for node2 in self.node_list:
                if node1 == node2: continue
                if graph.has_edge(node1.id, node2.id):
                    m += graph.edges[node1.id, node2.id]['value']
                    if w.get(node1.id) is None:
                        w[node1.id] = graph.edges[node1.id, node2.id]['value']
                    else:
                        w[node1.id] += graph.edges[node1.id, node2.id]['value']
        for node1 in self.node_list:
            for node2 in self.node_list:
                if node1 == node2: continue
                if graph.has_edge(node1.id, node2.id):
                    result += graph.edges[node1.id, node2.id]['value'] - w[node1.id] * w[node2.id] // m
                else:
                    result -= w[node1.id] * w[node2.id] // m
        # print(self.edge_index)
        # print(result / m)
        return result / m

    def update(self):
        for x in self.node_list:
            self.importance += x.importance
            for y in self.node_list:
                self.similarity += F.cosine_similarity(x.feature, y.feature, dim=1)
        self.importance /= len(self.node_list)
        self.similarity /= len(self.node_list) * len(self.node_list)

    def PYG_data(self):
        x = None
        for val in self.node_list:
            if x is None:
                x = torch.zeros_like(val.feature)
            else:
                x = torch.cat((x, val.feature))
        edge_index = torch.tensor(self.edge_index, dtype=torch.int64)
        edge_attr = torch.tensor(self.edge_attr, dtype=torch.float32)
        edge_weight = torch.tensor(self.edge_weight, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_wight=edge_weight)

    def has_edge(self, graph, other):
        for x in self.node_list:
            for y in other.node_list:
                if x == y or graph.has_edge(x, y):
                    return True
        return False

    def adaption(self, node, max_similarity, beta, graph):
        self.modularity(graph)
        temp = copy(self.node_list)
        temp.append(node)
        similarity = 0
        for x in temp:
            for y in temp:
                similarity += F.cosine_similarity(x.feature, y.feature, dim=1)
        similarity /= len(temp) * len(temp)
        if similarity > max_similarity:
            return False
        if self.similarity <= beta * similarity:
            self.similarity = similarity
            return True
        else:
            return False

    def __repr__(self):
        return "importance:{}".format(self.importance)


class node:
    def __init__(self, item_id, feature, importance):
        self.id = item_id
        self.feature = feature
        self.importance = importance


class user:
    def __init__(self, user_id, feature):
        self.feature = feature
        self.intention_feature = None
        self.intention_edge = None
        self.origin_PYG_data = None
        self.user_id = user_id
        self.item_list = list()
        self.intention_list = list()
        self.aggr_feature = None

    def user_origin_graph(self, graph, nodes, links):
        tensor = None
        st_list = list()
        ed_list = list()
        edge_attr = list()
        for x in links:
            if x.user == self.user_id:
                self.item_list.append(x.item)

        for x in self.item_list:
            if tensor is None:
                tensor = torch.zeros_like(nodes[x])
            tensor = torch.cat((tensor, nodes[x]))

        if len(self.item_list) == 0: return None
        for idx, x in enumerate(self.item_list):
            for idx2, y in enumerate(self.item_list):
                if graph.has_edge(x, y):
                    edge_attr.append([graph.edges[(x, y)]['value']])
                    st_list.append(idx + 1)
                    ed_list.append(idx2 + 1)

        for idx, x in enumerate(self.item_list):
            st_list.append(idx + 1)
            ed_list.append(0)

        st_list = torch.tensor(st_list, dtype=torch.int32)
        ed_list = torch.tensor(ed_list, dtype=torch.int32)
        st_list = torch.unsqueeze(st_list, 0)
        ed_list = torch.unsqueeze(ed_list, 0)
        edge_index = torch.cat((st_list, ed_list)).to(torch.int64)
        edge_attr = torch.tensor(edge_attr)
        index = torch.ones(tensor.shape[0], dtype=torch.long)
        self.origin_PYG_data = Data(x=tensor, edge_index=edge_index, edge_attr=edge_attr, index=index)

    def user_intention_graph(self, graph, nodes, links, max_intention=300, adapt=True, hop=1, beta=1, max_adaption=6):
        length = len(self.item_list)
        if length == 0: return None
        for i in range(length):
            for j in range(i + 1, length):
                for k in range(j + 1, length):
                    node1 = self.item_list[i]
                    node2 = self.item_list[j]
                    node3 = self.item_list[k]
                    if graph.has_edge(node1, node2) and graph.has_edge(node2, node3) and graph.has_edge(node1, node3):
                        node_a = node(node1, nodes[node1], graph.nodes[node1]['importance'])
                        node_b = node(node2, nodes[node2], graph.nodes[node2]['importance'])
                        node_c = node(node3, nodes[node3], graph.nodes[node3]['importance'])
                        edge_ab = graph.edges[(node1, node2)]['value']
                        edge_ac = graph.edges[(node1, node3)]['value']
                        edge_bc = graph.edges[(node2, node3)]['value']
                        self.intention_list.append(
                            intention([node_a, node_b, node_c], [[0, 0, 1], [1, 2, 2]],
                                      [[edge_ab], [edge_ac], [edge_bc]],
                                      [edge_ab, edge_ac, edge_bc]))
        # self.intention_list = random.sample(self.intention_list, min(max_intention, len(self.intention_list)))
        self.intention_list = sorted(self.intention_list, key=lambda intention: intention.importance, reverse=True)
        self.intention_list = self.intention_list[:min(max_intention, len(self.intention_list))]

        if adapt:
            max_similarity = 0
            for val in self.intention_list:
                max_similarity = max(max_similarity, val.similarity)
            for val in self.intention_list:
                for _ in range(hop):
                    if len(val.node_list) > max_adaption: break
                    node_list = copy(val.node_list)
                    for val2 in node_list:
                        if len(val.node_list) > max_adaption: break
                        for val3 in graph.neighbors(val2.id):
                            if len(val.node_list) > max_adaption: break
                            node_temp = node(val3, nodes[val3], graph.nodes[val3]['importance'])
                            flag = False
                            for f in val.node_list:
                                if f.id == val3:
                                    flag = True
                                    break
                            if flag: continue
                            if val.adaption(node_temp, max_similarity, beta, graph):
                                # cnt += 1
                                val.node_list.append(node_temp)
                                node_idx = len(val.node_list) - 1
                                if val3 not in self.item_list:
                                    self.item_list.append(val3)
                                for idx, x in enumerate(val.node_list):
                                    if graph.has_edge(x.id, node_temp.id):
                                        val.edge_index[0].append(idx)
                                        val.edge_index[1].append(node_idx)
                                        val.edge_attr.append([graph.edges[(x.id, node_temp.id)]['value']])
                                        val.edge_weight.append(graph.edges[(x.id, node_temp.id)]['value'])
        # for val in self.intention_list:
        #     print(len(val.node_list), end=" ")
        st_list = []
        ed_list = []
        for i in range(len(self.intention_list)):
            for j in range(i + 1, len(self.intention_list)):
                if self.intention_list[i].has_edge(graph, self.intention_list[j]):
                    st_list.append(i + 1)
                    ed_list.append(j + 1)
        for i in range(len(self.intention_list)):
            st_list.append(i + 1)
            ed_list.append(0)
        st_list = torch.tensor(st_list, dtype=torch.int32)
        ed_list = torch.tensor(ed_list, dtype=torch.int32)
        st_list = torch.unsqueeze(st_list, 0)
        ed_list = torch.unsqueeze(ed_list, 0)
        self.intention_edge = torch.cat((st_list, ed_list)).to(torch.int64)

    def get_intention_PYG(self):
        x = None
        for val in self.intention_list:
            if x is None:
                x = torch.zeros_like(val.feature)
            x = torch.cat((x, val.feature))
        index = torch.ones(x.shape[0], dtype=torch.long)
        return Data(x, self.intention_edge, index=index)


def build_L2L_graph(dataset='movielen'):
    nodes, users, links = get_from_csv(dataset)
    sorted(links)
    graph = nx.Graph()

    pre_item = dict()
    for x in links:
        if pre_item.get(x.user) is None:
            pre_item[x.user] = x.item
            continue
        pre = pre_item[x.user]
        pre_item[x.user] = x.item
        if graph.has_edge(pre, x.item):
            graph.edges[(pre, x.item)]['value'] += 1
        else:
            graph.add_edge(pre, x.item, value=1)

    pagerank = nx.pagerank(graph)
    for node in graph.nodes:
        graph.add_node(node, importance=pagerank[node])
    return nodes, users, links, graph


def random_choice(vis, array):
    result = choice(array)
    while vis.get(result) is not None:
        result = choice(array)
    return result


def get_random_intention(intention_size=10, dataset='yelp', hop=2, beta=1, max_adaption=6):
    nodes, users, links, graph = build_L2L_graph(dataset=dataset)
    # print("start neg")
    intention_list = []
    while True:
        node1 = random.sample(graph.nodes, 1)[0]
        if len(intention_list) > intention_size: break
        for node2 in graph.neighbors(node1):
            if len(intention_list) > intention_size: break
            for node3 in graph.neighbors(node1):
                if len(intention_list) > intention_size: break
                if node1 == node2: continue
                if graph.has_edge(node3, node2):
                    node_a = node(node1, nodes[node1], graph.nodes[node1]['importance'])
                    node_b = node(node2, nodes[node2], graph.nodes[node2]['importance'])
                    node_c = node(node3, nodes[node3], graph.nodes[node3]['importance'])
                    edge_ab = graph.edges[(node1, node2)]['value']
                    edge_ac = graph.edges[(node1, node3)]['value']
                    edge_bc = graph.edges[(node2, node3)]['value']
                    intention_list.append(intention([node_a, node_b, node_c], [[0, 0, 1], [1, 2, 2]],
                                                    [[edge_ab], [edge_ac], [edge_bc]],
                                                    [edge_ab, edge_ac, edge_bc]))

    max_similarity = 0
    for val in intention_list:
        max_similarity = max(max_similarity, val.similarity)
    for val in intention_list:
        for _ in range(hop):
            if len(val.node_list) > max_adaption: break
            node_list = copy(val.node_list)
            for val2 in node_list:
                if len(val.node_list) > max_adaption: break
                for val3 in graph.neighbors(val2.id):
                    if len(val.node_list) > max_adaption: break
                    node_temp = node(val3, nodes[val3], graph.nodes[val3]['importance'])
                    flag = False
                    for f in val.node_list:
                        if f.id == val3:
                            flag = True
                            break
                    if flag: continue
                    if val.adaption(node_temp, max_similarity, beta, graph):
                        val.node_list.append(node_temp)
                        node_idx = len(val.node_list) - 1
                        for idx, x in enumerate(val.node_list):
                            if graph.has_edge(x.id, node_temp.id):
                                val.edge_index[0].append(idx)
                                val.edge_index[1].append(node_idx)
                                val.edge_attr.append([graph.edges[(x.id, node_temp.id)]['value']])
                                val.edge_weight.append(graph.edges[(x.id, node_temp.id)]['value'])
    return intention_list


def get_user(train=True, max_intention=300, min_intention=10, dataset="yelp", hop=2, beta=1, max_adaption=6):
    nodes, users, links, graph = build_L2L_graph(dataset=dataset)
    user_list = list(users.keys())
    if train:
        user_list = user_list[:int(len(user_list) * 0.75)]
    else:
        user_list = user_list[int(len(user_list) * 0.75):]
    user_id = random.randint(1, len(user_list))
    now = user(user_id, users[user_id])
    now.user_origin_graph(graph, nodes, links)
    now.user_intention_graph(graph, nodes, links, max_intention=max_intention, hop=hop, beta=beta,
                             max_adaption=max_adaption)
    while now.intention_list is None or len(now.intention_list) < min_intention:
        # print(len(now.intention_list))
        user_id = random.randint(1, len(user_list))
        now = user(user_id, users[user_id])
        now.user_origin_graph(graph, nodes, links)
        now.user_intention_graph(graph, nodes, links, max_intention=max_intention, hop=hop, beta=beta,
                                 max_adaption=max_adaption)
    return now


def get_userlist(size, train=True, max_intention=300, min_intention=10, dataset='movielen', hop=2, beta=1,
                 max_adaption=6):
    nodes, users, links, graph = build_L2L_graph(dataset)
    # print("start pos")
    result = list()
    vis = dict()
    user_list = list(users.keys())
    if train:
        user_list = user_list[:int(len(user_list) * 0.75)]
    else:
        user_list = user_list[int(len(user_list) * 0.75):]
    for _ in range(size):
        user_id = random_choice(vis, user_list)
        vis[user_id] = 1
        now = user(user_id, users[user_id])
        now.user_origin_graph(graph, nodes, links)
        now.user_intention_graph(graph, nodes, links, max_intention=max_intention, hop=hop, beta=beta,
                                 max_adaption=max_adaption)
        while now.intention_list is None or len(now.intention_list) < min_intention:
            # print(len(now.intention_list))
            user_id = random_choice(vis, user_list)
            vis[user_id] = 1
            now = user(user_id, users[user_id])
            now.user_origin_graph(graph, nodes, links)
            now.user_intention_graph(graph, nodes, links, max_intention=max_intention, hop=hop, beta=beta,
                                     max_adaption=max_adaption)
        result.append(now)
    return result


if __name__ == '__main__':
    for x in get_random_intention(dataset="movielen", intention_size=10):
        print(len(x.node_list))
    # get_user(dataset="amazon")
    # get_userlist(10, dataset="amazon")
    # intentions = get_random_intention()
    # print(intentions)
