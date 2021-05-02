import torch.nn as nn
import torch

from utils.torch_utils import get_flat_params_from, set_flat_params_to

mseloss = nn.MSELoss()


class Worker:
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.local_iters = args['local_iters']
        self.gpu = args['gpu']
        self.secure = args['secure']
        self.clipping = args['clipping']
        if self.clipping == 1:
            self.train_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.clip = args['secure_clip'] 
        self.test_criterion = nn.CrossEntropyLoss()

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def train(self, train_dataloader, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        
        for iter in range(self.local_iters):
            train_loss = train_acc = train_total = 0
            count = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                count += y.size(0)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.train_criterion(pred, y)
                if self.clipping == 1:
                    saved_var = dict()
                    for tensor_name, tensor in self.model.named_parameters():
                        saved_var[tensor_name] = torch.zeros_like(tensor)
                    
                    for j in loss:
                        j.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                        for tensor_name, tensor in self.model.named_parameters():
                            new_grad = tensor.grad
                            saved_var[tensor_name].add_(new_grad)
                        self.model.zero_grad()
                        
                    for tensor_name, tensor in self.model.named_parameters():
                        tensor.grad = saved_var[tensor_name] / loss.shape[0]
                elif self.clipping == 2:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                
                if self.clipping == 1:
                    train_loss += torch.mean(loss).item() * y.size(0)
                else:
                    train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_soln = self.get_flat_model_params()
        stat_dict = {"loss": train_loss / train_total,
                     "acc": train_acc / train_total,
                     "subsample_size": count}

        return local_soln, stat_dict

    def test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                loss = self.test_criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss
