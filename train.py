import os
import yaml
from matplotlib import pyplot as plt
from torch.optim import Adam
import torch
from IaGELpy import net
from occurrence_graph import *


def load_args(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    try:
        args = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        args = yaml.load(f)
    return args


def train(origin=0):
    args = load_args("args.yaml")
    path = args['path']
    if args['model'] is None:
        input_dim = 845
        if args['dataset'] == 'movielen':
            input_dim = 20
        elif args['dataset'] == 'amazon':
            input_dim = 5967
        print("num of feature:{}".format(input_dim))
        model = net(input_dim, 128, 64, args['operator'],
                    args['dropout'], args['alpha']).to(args['device'])
        path = "save/{}/{}/{}_{}_{}".format(args['dataset'], args['operator'],
                                            args['alpha'], args['beta'], args['hop'])
        os.makedirs(path, exist_ok=True)

    else:
        model = torch.load("{}/{}".format(args['path'], args['model'])).to(args['device'])

    optim = Adam(model.parameters(), lr=args["learning_rate"])
    min_loss = None
    loss_list = list()
    model.train()
    for i in range(args['epoch']):
        optim.zero_grad()
        print("----------第{}轮训练开始-----------".format(i + 1 + origin))
        print("model ={},batch_size={},pos_intention={},neg_intention={},dataset={},hop={},alpha={},beta={},max={}"
              .format(args['operator'], args['batch_size'], args['pos_intention'], args['neg_intention'],
                      args['dataset'], args['hop'], args['alpha'], args['beta'], args['max_intention']))
        user_list = get_userlist(args['batch_size'], max_intention=args['pos_intention'],
                                 min_intention=args['pos_intention'], dataset=args['dataset'], hop=args['hop'],
                                 beta=args['beta'], max_adaption=args['max_intention'])
        intention_list = get_random_intention(intention_size=args['neg_intention'],
                                              dataset=args['dataset'], hop=args['hop'],
                                              beta=args['alpha'], max_adaption=args['max_intention'])
        loss = model(user_list, intention_list, args['device'])
        loss.backward()
        optim.step()
        print("训练轮数:{},Loss:{}".format(i + 1, loss.item()))
        loss_list.append(loss.cpu().detach().numpy())
        if (i + 1) % 50 == 0:
            # min_loss = loss
            torch.save(model, "{}/epoch_{}_batch_{}_sample_{}_max_{}".
                       format(path, i + 1 + origin, args['batch_size'], args['pos_intention'], args['max_intention']))

    show_loss(loss_list, args['pic_path'],
              "model ={},hop={},batch_size={},pos_intention={},neg_intention={},epoch={},dataset={},alpha={},beta={}".
              format(args['operator'], args['hop'], args['batch_size'], args['pos_intention'], args['neg_intention'],
                     args['epoch'] + origin, args['dataset'], args['alpha'], args['beta']), origin)


def show_loss(loss_list, path, label, origin):
    idx = list(range(1 + origin, len(loss_list) + 1 + origin))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.plot(idx, loss_list, label=label)
    plt.savefig("{}.jpg".format(os.path.join(path, label)))
    plt.show()


if __name__ == '__main__':
    train(0)
