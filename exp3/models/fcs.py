import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from scipy.spatial.distance import cdist
import time
from models.base import BaseLearner
from utils.inc_net import FCSNet
from utils.toolkit import count_parameters, tensor2numpy
from torch.nn import Parameter
from torch.optim.lr_scheduler import MultiStepLR

EPSILON = 1e-8

########################################
# =========== SupContrastive  ==========
########################################
class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
        
########################################
# =========== 多核 MMD函数  ============
########################################
def gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算多核Gaussian核的核矩阵 (x,y).
    x: [Nx, D], y: [Ny, D]
    kernel_mul: 对核带宽做指数扩展的基数
    kernel_num: 多个核
    fix_sigma: 若不为None, 使用固定sigma, 否则自动估计
    """
    n = x.size(0)
    m = y.size(0)
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (total.size(0)**2 - total.size(0))

    bandwidth /= (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)

def mmd_loss_func(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    多核MMD损失
    """
    batch_size = x.size(0)
    kernels = gaussian_kernel_matrix(x, y, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def rff_features(x, omega, b):
    """
    使用随机傅里叶特征将输入x映射到RFF空间.
    x: [N, D]；omega: [D_rff, D]；b: [D_rff]
    返回: [N, D_rff]
    """
    D_rff = omega.size(0)
    # sqrt(2/D_rff) 为缩放因子
    return torch.sqrt(torch.tensor(2.0 / D_rff, device=x.device)) * torch.cos(torch.matmul(x, omega.t()) + b)

def rff_mmd_loss_func(x, y, omega, b):
    """
    使用随机傅里叶特征（RFF）近似高斯核后计算 MMD 损失：
      loss = || mean(z(x)) - mean(z(y)) ||^2
    x: [N, D], y: [M, D]
    """
    z_x = rff_features(x, omega, b)  # [N, D_rff]
    z_y = rff_features(y, omega, b)  # [M, D_rff]
    diff = torch.mean(z_x, dim=0) - torch.mean(z_y, dim=0)
    return torch.sum(diff**2)

####################################################
# ============ FCS 主类: 引入 CVAE 回放 ============ #
####################################################
class FCS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FCSNet(args, False)

        self._protos = []
        self._covs = []
        self._radiuses = []
        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1

        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        self.log_dir = self.args["log_dir"]
        self.logs_name = "{}/{}/{}/{}/{}".format(
            args["model_name"], args["dataset"], init_cls, args["increment"], args["log_name"]
        )
        self.logs_name = os.path.join(self.log_dir, self.logs_name)

        self.contrast_loss = SupContrastive()
        # encoder_k 用于对比损失
        self.encoder_k = FCSNet(args, False).convnet
        
        # 记录当前epoch, 用于动态调整MMD
        self._current_epoch = 0
        self._epoch_num = args["epochs"]

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

        ckpt_name = "{}_{}_{}".format(self.args["model_name"], self.args["init_cls"], self.args["increment"])
        self.save_checkpoint(os.path.join(self.logs_name, ckpt_name))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        task_size = self.data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size

        # 更新分类头
        self._network.update_fc(
            self._known_classes * 4,
            self._total_classes * 4,
            int((task_size - 1) * task_size / 2)
        )
        self._network_module_ptr = self._network

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train',
            appendent=self._get_memory(),
            args=self.args
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"]
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def copy_state_dict(state_dict, model, strip=None):
        """
        如果有需要，保留你的旧逻辑。
        """
        tgt_state = model.state_dict()
        copied_names = set()
        for name, param in state_dict.items():
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
            if name not in tgt_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            if param.size() != tgt_state[name].size():
                print('mismatch:', name, param.size(), tgt_state[name].size())
                continue
            tgt_state[name].copy_(param)
            copied_names.add(name)
        missing = set(tgt_state.keys()) - copied_names
        if len(missing) > 0:
            print("missing keys in state_dict:", missing)
        return model

    def _train(self, train_loader, test_loader):
        resume = False

        if self._cur_task in range(self.args["ckpt_num"]):
            p = self.args["ckpt_path"]
            detail = p.split('/')
            l = "{}_{}_{}_{}.pkl".format('fcs', detail[-3], detail[-2], self._cur_task)
            l = os.path.join(p, l)
            print('load from {}'.format(l))
            self._network.load_state_dict(torch.load(l)["model_state_dict"], strict=False)
            resume = True

        self._network.to(self._device)

        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module

        if not resume:
            if self._cur_task == 0 and self.args["dataset"] == "imagenetsubset":
                # 特殊情况
                self._epoch_num = self.args["epochs_init"]
                print('use {} optimizer'.format(self._cur_task))
                base_lr = 0.1
                lr_strat = [80, 120, 150]
                lr_factor = 0.1
                custom_weight_decay = 5e-4
                custom_momentum = 0.9
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=base_lr,
                    momentum=custom_momentum,
                    weight_decay=custom_weight_decay
                )
                scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)
            else:
                self._epoch_num = self.args["epochs"]
                # 拆分CVAE参数的示例(可选)
                base_params = []
                cvae_params = []
                for n, p in self._network.named_parameters():
                    if 'cvae' in n and p.requires_grad:
                        cvae_params.append(p)
                    elif p.requires_grad:
                        base_params.append(p)

                optimizer = torch.optim.Adam([
                    {'params': base_params, 'lr': self.args["lr"]},
                    {'params': cvae_params, 'lr': self.args.get("lr_cvae", self.args["lr"])}
                ], weight_decay=self.args["weight_decay"])

                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"]
                )
            self._train_function(train_loader, test_loader, optimizer, scheduler)

        self._build_protos()

    def _build_protos(self):
        if self._cur_task != 0:
            proto = torch.tensor(self._protos).float().cuda()
            self._network.transfer.eval()
            with torch.no_grad():
                proto_transfer = self._network.transfer(proto)["logits"].cpu().tolist()
            self._network.transfer.train()
            for i in range(len(self._protos)):
                self._protos[i] = np.array(proto_transfer[i])

        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx+1),
                    source='train',
                    mode='test',
                    ret_data=True
                )
                idx_loader = DataLoader(
                    idx_dataset,
                    batch_size=self.args["batch_size"],
                    shuffle=False,
                    num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)

                cov = np.cov(vectors.T)
                self._covs.append(cov)
                self._radiuses.append(np.trace(cov)/vectors.shape[1])
            self._radius = np.sqrt(np.mean(self._radiuses))

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        start_time = time.time()
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch_idx in enumerate(prog_bar):
            self._current_epoch = epoch_idx + 1
            self._network.train()
            losses_sum = 0.
            losses_dict = {
                "loss_clf": 0., "loss_fkd": 0., "loss_proto": 0.,
                "loss_transfer": 0., "loss_contrast": 0., "loss_cvae": 0., "loss_mmd": 0.
            }
            correct, total = 0, 0

            for i, instance in enumerate(train_loader):
                # instance = (img_idx, inputs, targets, inputs_aug)
                (_, inputs, targets, inputs_aug) = instance
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                inputs_aug = inputs_aug.to(self._device, non_blocking=True)

                # 原有的 _class_aug 处理
                inputs, targets, inputs_aug = self._class_aug(inputs, targets, inputs_aug=inputs_aug)

                # 计算所有损失
                logits, losses_all = self._compute_il2a_loss(inputs, targets, image_k=inputs_aug)

                # 合并
                total_loss = 0.
                for k, v in losses_all.items():
                    total_loss += v
                    losses_dict[k] += v.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                losses_sum += total_loss.item()

                # 统计准确率
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            # 打印日志: 每个 epoch 的平均损失
            for k in losses_dict:
                losses_dict[k] /= len(train_loader)
            msg = (
                f"Task {self._cur_task}, Epoch {epoch_idx+1}/{self._epoch_num} => "
                f"Loss {losses_sum/len(train_loader):.3f}, "
                f"Loss_clf {losses_dict['loss_clf']:.3f}, "
                f"Loss_fkd {losses_dict['loss_fkd']:.3f}, "
                f"Loss_proto {losses_dict['loss_proto']:.3f}, "
                f"Loss_transfer {losses_dict['loss_transfer']:.3f}, "
                f"Loss_contrast {losses_dict['loss_contrast']:.3f}, "
                f"Loss_cvae {losses_dict['loss_cvae']:.3f}, "
                f"Loss_mmd {losses_dict['loss_mmd']:.3f}, "
                f"Train_accy {train_acc:.2f}"
            )

            if epoch_idx % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                msg += f", Test_accy {test_acc:.2f}"

            prog_bar.set_description(msg)
            logging.info(msg)
            avg_time = (time.time() - start_time) / self._current_epoch
            max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
            logging.info(f"Little Epoch Average Time per Epoch: {avg_time:.2f}s, Max GPU Memory: {max_mem:.2f}MB")
        end_time = time.time()
        avg_time = (end_time - start_time) / self._epoch_num
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        logging.info(f"Average Time per Epoch: {avg_time:.2f}s, Max GPU Memory: {max_mem:.2f}MB")

    #########################################
    # ============= CVAE + 多核MMD + 动态调控整合 (RFF版) ============ #
    #########################################
    def _compute_il2a_loss(self, inputs, targets, image_k=None):
        """
        GFR-FCS: 在原FCS基础上整合 CVAE (旧特征回放) + 对比学习 
        """
        # 先做一次前向
        network_output = self._network(inputs)
        features = network_output["features"]
        logits = network_output["logits"]

        # 一些默认损失初始化
        loss_clf = torch.tensor(0., device=self._device)
        loss_fkd = torch.tensor(0., device=self._device)
        loss_proto = torch.tensor(0., device=self._device)
        loss_transfer = torch.tensor(0., device=self._device)
        loss_contrast = torch.tensor(0., device=self._device)
        loss_cvae = torch.tensor(0., device=self._device)
        loss_mmd = torch.tensor(0., device=self._device)

        # 分类损失
        loss_clf = F.cross_entropy(logits/self.args["temp"], targets)

        # 第一阶段没有旧模型, 直接返回
        if self._cur_task == 0:
            return logits, {
                "loss_clf": loss_clf, "loss_fkd": loss_fkd,
                "loss_proto": loss_proto, "loss_transfer": loss_transfer,
                "loss_contrast": loss_contrast, "loss_cvae": loss_cvae,
                "loss_mmd": loss_mmd
            }

        # 取旧模型特征
        features_old = self.old_network_module_ptr.extract_vector(inputs)

        # =============== CVAE 生成损失 + 合成特征对比 ===============
        # 1) 训练CVAE: 计算 CVAE 损失 (重构 + KL)
        c_dim = self._network.total_classes
        condition = F.one_hot(targets, num_classes=c_dim).float().to(self._device)

        recon_feats, mu, logvar = self._network.cvae(features_old, condition)
        loss_recon = F.mse_loss(recon_feats, features_old.detach())
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta_cvae = self.args.get("beta_cvae", 0.1)
        loss_cvae = loss_recon + beta_cvae * loss_kl
        
        # 2) 多核 MMD 正则部分，使用 RFF 近似高斯核
        lambda_mmd_base = self.args.get("lambda_mmd_base", 50.0)
        factor = 1.0 - float(self._current_epoch)/(self._epoch_num + 1e-6)
        weight_mmd = lambda_mmd_base * max(factor, 0.)

        x_mmd = features
        y_mmd = features_old

        # 生成 RFF 的随机参数 omega 和 b
        D_rff = self.args.get("D_rff", 512)  # 设定随机特征数量
        sigma = self.args.get("sigma", 1.0)   # 高斯核的 sigma 参数
        d = x_mmd.size(1)
        # omega 的形状为 [D_rff, d]，从正态分布采样，标准差为1/sigma
        omega = torch.randn(D_rff, d, device=self._device) / sigma
        # b 的形状为 [D_rff]，均匀采样于 [0, 2pi)
        b = 2 * np.pi * torch.rand(D_rff, device=self._device)

        loss_mmd_val = rff_mmd_loss_func(x_mmd, y_mmd, omega, b)
        loss_mmd = weight_mmd * loss_mmd_val

        # =============== 其余已有损失: Transfer, FKD, Proto, Contrastive 等 ===============
        # Transfer Loss
        feature_transfer = self._network.transfer(features_old)["logits"]
        loss_transfer = self.args["lambda_transfer"] * self.l2loss(features, feature_transfer)

        # FKD
        loss_fkd = self.args["lambda_fkd"] * self.l2loss(features, features_old, mean=False)

        # Proto
        index = np.random.choice(range(self._known_classes), size=self.args["batch_size"], replace=True)
        proto_features_raw = np.array(self._protos)[index]
        proto_targets = index * 4
        proto_features = proto_features_raw + np.random.normal(0, 1, proto_features_raw.shape)*self._radius
        proto_features = torch.from_numpy(proto_features).float().to(self._device)
        proto_targets = torch.from_numpy(proto_targets).to(self._device)
        proto_transfer = self._network.transfer(proto_features)["logits"].detach()
        proto_logits = self._network_module_ptr.fc(proto_transfer)["logits"][:, : self._total_classes*4]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets)
        
        # 对比损失 (和原有一致)
        if image_k is not None and (self._cur_task > 0):
            b = image_k.shape[0]
            targets_part = targets[:b].clone()
            with torch.no_grad():
                self._copy_key_encoder()
                feats_k = self.encoder_k(image_k)["features"]    # shape=[b, D]
                feats_k = torch.cat((feats_k, proto_features), dim=0) # shape=[b+proto_count, D]
                feats_k = F.normalize(feats_k, dim=-1)
            feats_q = F.normalize(features[:b], dim=-1)
            l_pos = (feats_q * feats_k[:b]).sum(-1).view(-1, 1)
            l_neg = torch.einsum('nc,ck->nk', [feats_q, feats_k.T])
            logits_global = torch.cat([l_pos, l_neg], dim=1)

            positive_target = torch.ones((b, 1), device=self._device)
            all_targets_k = torch.cat([targets_part, proto_targets], dim=0)  # shape=[b+proto_count]
            negative_targets = (targets_part.unsqueeze(1) == all_targets_k.unsqueeze(0)).float()
            targets_global = torch.cat([positive_target, negative_targets], dim=1)
            loss_contrast = self.contrast_loss(logits_global, targets_global)*self.args["lambda_contrast"]

        # 最后汇总
        losses_all = {
            "loss_clf": loss_clf,
            "loss_fkd": loss_fkd,
            "loss_proto": loss_proto,
            "loss_transfer": loss_transfer,
            "loss_contrast": loss_contrast,
            "loss_cvae": loss_cvae,
            "loss_mmd": loss_mmd
        }
        return logits, losses_all
        
    def l2loss(self, inputs, targets, mean=True):
        # 添加 1e-8 确保开方内部不为 0
        eps = 1e-8
        if mean:
            delta = torch.sqrt(torch.sum((inputs - targets)**2, dim=-1) + eps)
            return torch.mean(delta)
        else:
            delta = torch.sqrt(torch.sum((inputs - targets)**2) + eps)
            return delta
    
    @torch.no_grad()
    def _copy_key_encoder(self):
        """Momentum update of the key encoder"""
        self.encoder_k.to(self._device)
        for param_q, param_k in zip(self._network.convnet.parameters(), self.encoder_k.parameters()):
            param_k.data = param_q.data

    def _class_aug(self, inputs, targets, alpha=20., mix_time=4, inputs_aug=None):
        """
        你的原 Mixup + rotation 数据增强逻辑
        """
        # rotation
        inputs2 = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], dim=1)
        inputs2 = inputs2.view(-1, 3, inputs2.shape[-2], inputs2.shape[-1])
        targets2 = torch.stack([targets * 4 + k for k in range(4)], dim=1).view(-1)

        inputs_aug2 = torch.stack([torch.rot90(inputs_aug, k, (2, 3)) for k in range(4)], dim=1)
        inputs_aug2 = inputs_aug2.view(-1, 3, inputs_aug2.shape[-2], inputs_aug2.shape[-1])

        mixup_inputs = []
        mixup_targets = []

        for _ in range(mix_time):
            index = torch.randperm(inputs.shape[0])
            perm_inputs = inputs[index]
            perm_targets = targets[index]
            mask = perm_targets != targets

            select_inputs = inputs[mask]
            select_targets = targets[mask]
            perm_inputs = perm_inputs[mask]
            perm_targets = perm_targets[mask]

            lams = np.random.beta(alpha, alpha, sum(mask))
            lams = np.where((lams < 0.4) | (lams > 0.6), 0.5, lams)
            lams = torch.from_numpy(lams).float().to(self._device)[:, None, None, None]

            mixup_inputs.append(lams * select_inputs + (1 - lams) * perm_inputs)
            mixup_targets.append(self._map_targets(select_targets, perm_targets))

        mixup_inputs = torch.cat(mixup_inputs, dim=0)
        mixup_targets = torch.cat(mixup_targets, dim=0)

        inputs = torch.cat([inputs2, mixup_inputs], dim=0)
        targets = torch.cat([targets2, mixup_targets], dim=0)
        return inputs, targets, inputs_aug2

    def _map_targets(self, select_targets, perm_targets):
        assert (select_targets != perm_targets).all()
        large_targets = torch.max(select_targets, perm_targets) - self._known_classes
        small_targets = torch.min(select_targets, perm_targets) - self._known_classes
        mixup_targets = (large_targets*(large_targets-1)//2 + small_targets + self._total_classes*4).long()
        return mixup_targets

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, :self._total_classes*4][:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader, only_new, only_old):
        """
        如果需要评测 only_new/only_old
        """
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, :self._total_classes*4][:, ::4]
                if only_new:
                    outputs[:, :self._known_classes] = -100
                if only_old:
                    outputs[:, self._known_classes:] = -100
            predicts = torch.topk(outputs, k=self.topk, dim=1)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self, only_new=False, only_old=False):
        y_pred, y_true = self._eval_cnn(self.test_loader, only_new=only_new, only_old=only_old)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred_nme, y_true_nme = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred_nme, y_true_nme)
        elif hasattr(self, '_protos') and len(self._protos) == self._known_classes:
            # 以 _protos 当做 class_mean
            class_means = np.array(self._protos) / (np.linalg.norm(self._protos, axis=1, keepdims=True)+1e-8)
            y_pred_nme, y_true_nme = self._eval_nme(self.test_loader, class_means)
            nme_accy = self._evaluate(y_pred_nme, y_true_nme)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True)+EPSILON)
        dists = cdist(class_means, vectors, 'sqeuclidean')
        scores = dists.T
        y_pred = np.argsort(scores, axis=1)[:, :self.topk]
        return y_pred, y_true
