import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from model import ResNet3D
from reader import KineticsReader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tsn.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to last weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--pretrain',
        type=str,
        default='work/res50',
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        val_config = merge_configs(config, 'test', vars(args))

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        # 根据自己定义的网络，声明train_model
        train_model = ResNet3D.generate_model(50)

        if args.resume == True:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/tsn_model')
            train_model.load_dict(model)
            print('Resume from ' + args.save_dir + '/tsn_model')
        elif args.pretrain:
            pretrain_weights = fluid.io.load_program_state(args.pretrain)
            inner_state_dict = train_model.state_dict()
            print('Resume from' + args.pretrain)
            for name, para in inner_state_dict.items():
                if ((name in pretrain_weights) and (not ('fc' in para.name))):
                    para.set_value(pretrain_weights[name])
                else:
                    print('del ' + para.name)

        opt = fluid.optimizer.Momentum(train_config.TRAIN.learning_rate,
                                       train_config.TRAIN.learning_rate_decay,
                                       parameter_list=train_model.parameters())
        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()
        val_reader = KineticsReader(args.model_name.upper(), 'valid', val_config).create_reader()

        epochs = args.epoch or train_model.epoch_num()
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):  # data (list) (batch)[seg_num,3 * seglen,size,size]
                dy_x_data = np.array([x[0] for x in data]).astype('float32')  # [batch, seg_num, 3 * seglen, size, size]
                y_data = np.array([[x[1]] for x in data]).astype('int64')  # [batch, 1]

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = train_model(img, label)

                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()

                logger.info(
                    "Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))

            acc_list = []
            for batch_id, data in enumerate(val_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = train_model(img, label)
                acc_list.append(acc.numpy()[0])

            logger.info(
                "Val at epoch {}:  acc: {}".format(i, np.mean(acc_list)))
            print("Val at epoch {}:  acc: {}".format(i, np.mean(acc_list)) + '\n')

            if i % 10 == 0:
                fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/tsn_model_' + str(i))
        fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/tsn_model')
        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
