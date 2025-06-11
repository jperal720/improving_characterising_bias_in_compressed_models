import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PruningHelper import PruningHelper as ph
from torch.utils.data import DataLoader
from torch.nn.utils.prune import global_unstructured, L1Unstructured
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

class ModelUtils:
    def __init__(self, device):
        self.device = device


    def trainer(self, model, optimizer, ds_train, K_FOLDS, BATCH_SIZE, EPOCHS, pruning=False, pruning_schedule=[], pruning_rate=0):
        criterion = nn.BCEWithLogitsLoss()

        kf = KFold(n_splits=K_FOLDS, shuffle=True)

        total_loss = []
        total_accuracy = []
        total_precision = []
        total_recall = []
        total_f1_score = []

        # Training
        step = 0
        num_of_prunes = 0
        for fold, (train_k, val_k) in enumerate(kf.split(ds_train)):

            train_loader_k = DataLoader(dataset=ds_train, batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(train_k))

            val_loader_k = DataLoader(
                dataset=ds_train,
                batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SubsetRandomSampler(val_k)
            )

            model.train()
            for epoch in range(EPOCHS):
                epoch_loss = 0
                for imgs, lbls, _ in train_loader_k:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device).float()

                    outputs = model(imgs).squeeze(1)
                    loss = criterion(outputs, lbls)
                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()

                    # Add step
                    step+=1

                    # Pruning
                    if pruning and (step in pruning_schedule):
                        num_of_prunes += 1
                        global_unstructured(
                            [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))],
                            pruning_method=L1Unstructured,
                            amount=pruning_rate,
                        )
                        print(f"[Step {step}]Num of prunes -> {num_of_prunes}\nTotal sparsity: {ph.get_sparsity(model)}")


                    optimizer.step()
                    
                avg_epoch_loss = epoch_loss / len(train_loader_k)
                total_loss.append(avg_epoch_loss)
                print(f"Fold: {fold+1}, Epoch: [{epoch + 1}/{EPOCHS}], Loss: {avg_epoch_loss:.7f}")

            # Evaluation of model
            model.eval()
            total=0
            correct=0
            y_pred = []
            y_true = []

            with torch.no_grad():
                for imgs, lbls, _ in val_loader_k:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device).float()
                    outputs = model(imgs).squeeze(1)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()

                    y_true.extend(lbls.cpu().numpy()) #storing true values
                    y_pred.extend(predicted.cpu().numpy()) #storing predictions
                    total += lbls.size(0)
                    correct += (predicted == lbls).sum().item()


            accuracy = correct / total
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

            total_accuracy.append(accuracy)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1_score.append(f1)

            print(f"Fold: [{fold + 1}/{K_FOLDS}], Accuracy: {accuracy}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

        print(f"Avg. accuracy: {np.mean(total_accuracy):.2f}")
        print(f"Avg. precision: {np.mean(total_precision):.2f}")
        print(f"Avg. Recall: {np.mean(total_recall):.2f}")
        print(f"Avg. F1-score: {np.mean(total_f1_score):.2f}")

        print("Training Completed")
        print(f"Number of steps {step}")

        return model, total_loss, total_accuracy, total_precision, total_recall, total_f1_score

    def test_model(self, model, test_loader):
        '''
        Function for determining test scores of a model on a test set
        '''
        print("Test Evaluation")
    
        model.eval()
        y_pred_test = []
        y_true_test = []
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for imgs, lbls, _ in test_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device).float()
                outputs = model(imgs).squeeze(1)
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                y_true_test.extend(lbls.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())

                total_samples += lbls.size(0)
                correct_predictions += (predicted == lbls).sum().item()

        accuracy = correct_predictions / total_samples
        precision_test = precision_score(y_true_test, y_pred_test, average='binary', zero_division=0)
        recall_test = recall_score(y_true_test, y_pred_test, average='binary', zero_division=0)
        f1_score_test = f1_score(y_true_test, y_pred_test, average='binary',zero_division=0)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Test precision: {precision_test:.2f}")
        print(f"Test recall: {recall_test:.2f}")
        print(f"Test f1-score: {f1_score_test:.2f}")

        print("Finished test evaluation!")


    def plot_loss_curves(self, train_losses, K_FOLDS, EPOCHS):
        if K_FOLDS != 4:
            return

        fig, axes = plt.subplots(1,4, figsize=(10,3))
        axes = axes.ravel()
        total_losses = len(train_losses)
        losses_per_fold = total_losses / K_FOLDS #Losses included per graph

        for graph in range(K_FOLDS):
            start_loss = int(graph * losses_per_fold)
            end_loss = int(start_loss + losses_per_fold)
            losses = np.array(train_losses[start_loss:end_loss])
            epochs = np.arange(1, EPOCHS + 1)

            axes[graph].plot(epochs, losses, linewidth=2)
            axes[graph].set_title(f'Fold {graph+1}', fontsize=12)
            axes[graph].set_xlabel('Epochs', fontsize=10)
            axes[graph].set_ylabel('Loss', fontsize=10)
            axes[graph].grid(True, linestyle='--', alpha=0.5)

            # Setting y-limits
            y_padding = 0.1 * (np.max(train_losses) - np.min(train_losses))
            axes[graph].set_ylim([np.min(train_losses) - y_padding,
                                np.max(train_losses) + y_padding])
            
        plt.suptitle(f"{K_FOLDS} Cross Validation Loss Curves", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()



    def get_cie_count(self, model_1, model_2, test_loader):
        '''
        Function for determining modular CIEs between two models
        '''
        print("Determining CIEs:")

        model_1.eval()
        model_2.eval()
        cie_list = []

        with torch.no_grad():
            for imgs, lbls, sample_name in test_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)

                outputs_1 = model_1(imgs).squeeze(1)
                predicted_1 = (torch.sigmoid(outputs_1) > 0.5).float()

                outputs_2 = model_2(imgs).squeeze(1)
                predicted_2 = (torch.sigmoid(outputs_2) > 0.5).float()

                disagree_mask = (predicted_1 != predicted_2)

                for i in range(len(sample_name)):
                     if disagree_mask[i]:
                          cie_list.append(sample_name[i])

        print(f"Number of CIEs: {len(cie_list)}")

        return cie_list
    

    def show_cie_eda_unitary_plot(self, cie_list, anno_path: str):
        df = pd.read_csv(anno_path, sep=r"\s+", header=1)
        df.replace(-1, 0, inplace=True)
        df = df.loc[cie_list]

        # Guard clause for ensuring df loads proper values
        if(len(cie_list) != df.shape[0]):
            return

        # Gender
        male = df[df['Male'] == 1]
        female = df[df['Male'] == 0]

        # Age
        young = df[df['Young'] == 1]
        old = df[df['Young'] == 0]

        print(f"Young: {young.shape[0]}, Old: {old.shape[0]}, Females: {female.shape[0]}, Males: {male.shape[0]}")

        u_labels = [
            'Young', 'Old', 'Females', 'Males'
        ]
        u_counts = [
            young.shape[0], old.shape[0], female.shape[0], male.shape[0]
        ]

        plt.figure(figsize=(12, 6))
        plt.bar(u_labels, u_counts, color=['#FFC0CB', '#87CEEB', '#90EE90', '#FFD700'])
        plt.title('Unitary Attribute Breakdown in CIE')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def show_cie_eda_inter_plot(self, cie_list, anno_path: str):
        df = pd.read_csv(anno_path, sep=r"\s+", header=1)
        df.replace(-1, 0, inplace=True)
        df = df.loc[cie_list]

        # Guard clause for ensuring df loads proper values
        if(len(cie_list) != df.shape[0]):
            return

        # Gender
        male = df[df['Male'] == 1]
        female = df[df['Male'] == 0]

        # Intersecting attributes
        male_young = male[male['Young'] == 1].shape[0]
        male_old = male[male['Young'] == 0].shape[0]
        female_young = female[female['Young'] == 1].shape[0]
        female_old = female[female['Young'] == 0].shape[0]

        print(f"Young Females: {female_young}, Young Males: {male_young}, Old Males: {male_old}, Old Females: {female_old}")

        g_labels = [
            'Young Females', 'Young Males', 'Old Males', 'Old Females'
        ]
        g_counts = [
            female_young, male_young, male_old, female_old
        ]


        plt.figure(figsize=(12, 6))
        plt.bar(g_labels, g_counts, color=['#FFC0CB', '#87CEEB', '#90EE90', '#FFD700'])
        plt.title('2-Class Intersectional Attribute Breakdown in CIE')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


    def show_cie_eda_blonde_inter_plot(self, cie_list, anno_path: str):
        df = pd.read_csv(anno_path, sep=r"\s+", header=1)
        df.replace(-1, 0, inplace=True)
        df = df.loc[cie_list]

        # Guard clause for ensuring df loads proper values
        if(len(cie_list) != df.shape[0]):
            return

        # Blonde
        blonde = df[df["Blond_Hair"] == 1]
        non_blonde = df[df["Blond_Hair"] == 0]

        # Intersecting Blonde attributes
        blonde_female = blonde[blonde["Male"] == 0].shape[0]
        blonde_male = blonde[blonde["Male"] == 1].shape[0]
        blonde_young = blonde[blonde["Young"] == 1].shape[0]
        blonde_old = blonde[blonde["Young"] == 0].shape[0]

        #Intersecting Non-blonde attributes
        non_blonde_female = non_blonde[non_blonde["Male"] == 0].shape[0]
        non_blonde_male = non_blonde[non_blonde["Male"] == 1].shape[0]
        non_blonde_young = non_blonde[non_blonde["Young"] == 1].shape[0]
        non_blonde_old = non_blonde[non_blonde["Young"] == 0].shape[0]

        print(f"Non-Blonde Young: NBY\nNon-Blonde Female: NBF\nNon-Blonde Male: NBM\nNon-Blonde Old: NBO\nBlonde Female: BF\nBlonde Young: BY\nBlonde Old: BO\nBlonde Male: BM")

        print(f"NBY: {non_blonde_young}, NBF: {non_blonde_female}, NBM: {non_blonde_male}, NBO: {non_blonde_old}, BF: {blonde_female}, BY: {blonde_young}, BO: {blonde_old}, BM: {blonde_male}")

        b_labels = [
            'Non-Blonde Young', 'Non-Blonde Female', 'Non-Blonde Male', 'Non-Blonde Old', 'Blonde Female', 'Blonde Young', 'Blonde Old', 'Blonde Male' 
        ]
        b_counts = [
            non_blonde_young, non_blonde_female, non_blonde_male, non_blonde_old, blonde_female, blonde_young, blonde_old, blonde_male
        ]

        plt.figure(figsize=(12, 6))
        plt.bar(b_labels, b_counts, color=['#FFC0CB', '#87CEEB', '#90EE90', '#FFD700',
                                    '#FFB6C1', '#ADD8E6', '#98FB98', '#D3D3D3'])
        plt.title('Blonde and Non-Blonde Intersectional Attribute Breakdown in CIE')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()