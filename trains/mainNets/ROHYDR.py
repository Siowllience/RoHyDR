import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
logger = logging.getLogger('MMSA')
class ROHYDR():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.L1Loss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        optimizer_D = optim.Adam(model.discriminator.parameters(), lr=self.args.learning_rate, weight_decay=self.args.decay) 

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0


        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters before load: {total_params:,}")

        # To reproduce the results of the paper, you can load the pre-trained model here. Please write the path of the pre-trained model.
        origin_model = torch.load('pretrained/pretrained-{}.pth'.format(self.args.dataset_name))
        net_dict = model.state_dict()
        new_state_dict = {}
        for k, v in origin_model.items():
            k = k.replace('Model.', '')
            new_state_dict[k] = v
        net_dict.update(new_state_dict)
        model.load_state_dict(net_dict,strict=False)

        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters after load: {total_params:,}")

        # def count_parameters(model, filepath="/disk2/home/yuehan.jin/project/2025rebuttal/para_rohydr.txt"):
        #     with open(filepath, "w") as f:
        #         print(f"{'Module':<50} {'Params':>12}", file=f)
        #         print("="*65, file=f)
        #         total_params = 0
        #         for name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 num_params = param.numel()
        #                 total_params += num_params
        #                 print(f"{name:<50} {num_params:>12}", file=f)
        #         print("="*65, file=f)
        #         print(f"{'Total Trainable Params':<50} {total_params:>12,}", file=f)

        # count_parameters(model)


        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            miss_one, miss_two = 0, 0  # num of missing one modal and missing two modal
            left_epochs = self.args.update_epochs

            avg_rloss = []
            avg_closs = []
            avg_dloss = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    miss_2 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
                    miss_1 = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0]

                    if miss_two / (np.round(len(dataloader['train']) / 10) * 10) < miss_2[int(self.args.mr*10-1)]:  # missing two modal
                        missing_modal=1
                        miss_two += 1
                    elif miss_one / (np.round(len(dataloader['train']) / 10) * 10) < miss_1[int(self.args.mr*10-1)]:  # missing one modal
                        missing_modal=2
                        miss_one += 1
                    else:  # no missing
                        missing_modal=3
                    # missing_modal=2

                    #Optimization Stage1
                    if missing_modal!=3:
                        if missing_modal==1:  # missing two modal
                            outputs = model(text, audio, vision, num_modal=1)
                        elif missing_modal==2:  # missing one modal
                            outputs = model(text, audio, vision, num_modal=2)
                        else:  # no missing
                            outputs = model(text, audio, vision, num_modal=3)
                        loss_score_l = outputs['loss_score_l']
                        loss_score_v = outputs['loss_score_v']
                        loss_score_a = outputs['loss_score_a']
                        loss_rec = outputs['loss_rec']
                        S1_loss=((loss_score_l + loss_score_v + loss_score_a)+loss_rec * self.args.lambda_g) / (1 + self.args.lambda_g)
                        # S1_loss=loss_score_l + loss_score_v + loss_score_a + loss_rec
                        S1_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    

                    #Optimization Stage2
                    if missing_modal!=3:
                        if missing_modal==1:  # missing two modal
                            outputs_S2 = model(text, audio, vision, num_modal=1)
                        elif missing_modal==2:  # missing one modal
                            outputs_S2 = model(text, audio, vision, num_modal=2)
                        else:  # no missing
                            outputs_S2 = model(text, audio, vision, num_modal=3)

                        #(1):Discriminator
                        valid = torch.ones(size=[labels.shape[0], 1], requires_grad=False).type_as(audio).to(self.args.device)
                        fake = torch.zeros(size=[labels.shape[0], 1], requires_grad=False).type_as(audio).to(self.args.device)
                        optimizer_D.zero_grad()
                        fusion_feature_x = outputs_S2['Fusion_gt']
                        fusion_feature_lm = outputs_S2['Fusion_lm']
                        real_loss = self.adversarial_loss(model.discriminator(fusion_feature_x.clone().detach()), valid)
                        fake_loss = self.adversarial_loss(model.discriminator(fusion_feature_lm.clone().detach()), fake)
                    
                        d_loss = 0.1 * (real_loss + fake_loss)
                        avg_dloss.append(d_loss.item())
                        d_loss.backward()
                        optimizer_D.step()

                        #(2):Multimodal Reconstructor
                        optimizer.zero_grad()
                        recon_fusion_f = model.reconstruction(fusion_feature_lm)
                        rl1 = self.pixelwise_loss(fusion_feature_x, recon_fusion_f)
                        avg_rloss.append(rl1.item())
                        t = model.discriminator(recon_fusion_f)
                        advl1 = self.adversarial_loss(t, valid)
                        g_loss =(self.args.lambda_al * (advl1) + (1-self.args.lambda_al) * (rl1)) 
                        g_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    
                    #Optimization Stage3
                    if missing_modal==1:  # missing two modal
                        outputs_S3 = model(text, audio, vision, num_modal=1)
                    elif missing_modal==2:  # missing one modal
                        outputs_S3 = model(text, audio, vision, num_modal=2)
                    else:  # no missing
                        outputs_S3 = model(text, audio, vision, num_modal=3)
                    fusion_feature_x = outputs_S3['Fusion_gt']
                    if missing_modal==3:
                        fusion_feature_lm = outputs_S3['Fusion_lm']
                    else:
                        fusion_feature_lm=model.reconstruction(outputs_S3['Fusion_lm'])
                    classify_output=model.getoutput(fusion_feature_x,fusion_feature_lm)
                    # task_loss = (self.criterion(classify_output['M_gt'], labels)+self.criterion(classify_output['M_lm'], labels) * self.args.lambda_c) / (1 + self.args.lambda_c)
                    task_loss = (self.criterion(classify_output['M_gt'], labels)*(1-self.args.lambda_c)+self.criterion(classify_output['M_lm'], labels) * self.args.lambda_c)
                    combine_loss = task_loss
                    # backward
                    combine_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    # store results
                    # train_loss += combine_loss.item()
                    train_loss +=S1_loss.item()
                    y_pred.append(classify_output['M_lm'].cpu())
                    y_true.append(labels.cpu())
                    y_pred.append(classify_output['M_gt'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            model_save_path = 'pretrained/' + str(epochs) + '.pth'
            torch.save(model.state_dict(), model_save_path)
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        miss_one, miss_two = 0, 0

        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    miss_2 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
                    miss_1 = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0]
                    if miss_two / (np.round(len(dataloader) / 10) * 10) < miss_2[int(self.args.mr * 10 - 1)]:  # missing two modal
                        outputs = model(text, audio, vision, num_modal=1)
                        missing_modal=1
                        miss_two += 1
                    elif miss_one / (np.round(len(dataloader) / 10) * 10) < miss_1[int(self.args.mr * 10 - 1)]:  # missing one modal
                        outputs = model(text, audio, vision, num_modal=2)
                        missing_modal=2
                        miss_one += 1
                    else:  # no missing
                        missing_modal=3
                        outputs = model(text, audio, vision, num_modal=3)
                    
                    # outputs = model(text, audio, vision, num_modal=2)
                    # missing_modal=2
                    
                    fusion_feature_x = outputs['Fusion_gt']
                    if missing_modal==3:
                        fusion_feature_lm = outputs['Fusion_lm']
                    else:
                        fusion_feature_lm=model.reconstruction(outputs['Fusion_lm'])
                    classify_output=model.getoutput(fusion_feature_x,fusion_feature_lm)
                    loss = self.criterion(classify_output['M_lm'], labels)
                    eval_loss += loss.item()
                    y_pred.append(classify_output['M_lm'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        return eval_results