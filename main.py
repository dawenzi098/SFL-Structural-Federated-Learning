import torch
import random
import copy
import numpy as np
import time
from BResidual import BResidual
from options import arg_parameter
from data_util import load_cifar10, load_mnist
from federated import Cifar10FedEngine
from aggregator import parameter_aggregate, read_out
from util import *


def main(args):
    args.device = torch.device(args.device)

    print("Prepare data and model...")
    train_batches, test_batches, A, overall_tbatches = load_cifar10(args)
    model = BResidual(3)

    print("Parameter holders")
    w_server, w_local = model.get_state()
    w_server = [w_server] * args.clients
    w_local = [w_local] * args.clients
    global_model = copy.deepcopy(w_server)
    personalized_model = copy.deepcopy(w_server)

    server_state = None
    client_states = [None] * args.clients

    print2file(str(args), args.logDir, True)
    nParams = sum([p.nelement() for p in model.parameters()])
    print2file('Number of model parameters is ' + str(nParams), args.logDir, True)

    print("Start Training...")
    num_collaborator = max(int(args.client_frac * args.clients), 1)
    for com in range(1, args.com_round + 1):
        selected_user = np.random.choice(range(args.clients), num_collaborator, replace=False)
        train_time = []
        train_loss = []
        train_acc = []

        for c in selected_user:
            # Training
            engine = Cifar10FedEngine(args, copy.deepcopy(train_batches[c]), global_model[c], personalized_model[c],
                                          w_local[c], {}, c, 0, "Train", server_state, client_states[c])
            outputs = engine.run()

            w_server[c] = copy.deepcopy(outputs['params'][0])
            w_local[c] = copy.deepcopy(outputs['params'][1])
            train_time.append(outputs["time"])
            train_loss.append(outputs["loss"])
            train_acc.append(outputs["acc"])
            client_states[c] = outputs["c_state"]

        mtrain_time = np.mean(train_time)
        mtrain_loss = np.mean(train_loss)
        mtrain_acc = np.mean(train_acc)

        log = 'Communication Round: {:03d}, Train Loss: {:.4f},' \
            ' Train Accuracy: {:.4f}, Training Time: {:.4f}/com_round'
        print2file(log.format(com, mtrain_time, mtrain_loss, mtrain_acc),
                args.logDir, True)

        # Server aggregation
        t1 = time.time()
        personalized_model, client_states, server_state = \
            parameter_aggregate(args, A, w_server, global_model, server_state, client_states, selected_user)
        t2 = time.time()
        log = 'Communication Round: {:03d}, Aggregation Time: {:.4f} secs'
        print2file(log.format(com, (t2 - t1)), args.logDir, True)

        # Readout for global model
        global_model = read_out(personalized_model, args.device)

        # Validation
        if com % args.valid_freq == 0:
            single_vtime = []
            single_vloss = []
            single_vacc = []

            all_vtime = []
            all_vloss = []
            all_vacc = []

            for c in range(args.clients):
                batch_time = []
                batch_loss = []
                batch_acc = []

                for batch in test_batches:
                    tengine = Cifar10FedEngine(args, copy.deepcopy(batch), personalized_model[c], personalized_model[c],
                                               w_local[c], {}, c, 0, "Test", server_state, client_states[c])

                    outputs = tengine.run()

                    batch_time.append(outputs["time"])
                    batch_loss.append(outputs["loss"])
                    batch_acc.append(outputs["acc"])

                single_vtime.append(batch_time[c])
                single_vloss.append(batch_loss[c])
                single_vacc.append(batch_acc[c])

                all_vtime.append(np.mean(batch_time))
                all_vloss.append(np.mean(batch_loss))
                all_vacc.append(np.mean(batch_acc))


            single_log = 'SingleValidation Round: {:03d}, Valid Loss: {:.4f}, ' \
                  'Valid Accuracy: {:.4f}, Valid SD: {:.4f}, Test Time: {:.4f}/epoch'
            print2file(single_log.format(com, np.mean(single_vloss), np.mean(single_vacc), np.std(single_vacc),
                                         np.mean(single_vtime)), args.logDir, True)

            all_log = 'AllValidation Round: {:03d}, Valid Loss: {:.4f}, ' \
                         'Valid Accuracy: {:.4f}, Valid SD: {:.4f}, Test Time: {:.4f}/epoch'
            print2file(all_log.format(com, np.mean(all_vloss), np.mean(all_vacc), np.std(all_vacc),
                                      np.mean(all_vtime)), args.logDir, True)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    option = arg_parameter()
    initial_environment(option.seed)
    main(option)

    print("Everything so far so good....")