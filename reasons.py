import torch
import numpy as np
from torch.utils.data import DataLoader

class TorchDoubtLab:

    def __init__(self, model, test_dataset, batch_size=16):

        self.model = model
        self.dataset = test_dataset
        self.batch_size = batch_size

        self.preds, self.pred_confs, self.labels = self.get_preds()

        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

    def get_preds(self):

        preds = []
        pred_confs = []
        labels = []
        
        for image,label in self.test_loader:

            conf = self.model(image)
            #softmax normalize conf logit outputs
            conf_norm = torch.nn.functional.softmax(conf, dim=1)
            conf_results_np = conf_norm.cpu().detach().numpy().tolist()
            labels_np = label.cpu().detach().numpy().tolist()
            preds_np = np.argmax(conf_results_np, axis=1).tolist()

            preds += preds_np
            pred_confs += conf_results_np
            labels += labels_np

        return np.array(preds), np.array(pred_confs), np.array(labels)

    def get_flagged_image(self, flagged_idx):
        
        flagged_images = []
        for idx in flagged_idx:
            flagged_images.append(self.dataset.image_paths[idx])
            
        return flagged_images

    def ProbaReason(self, max_proba=0.55):

        flagged_idx = []

        for i, result in enumerate(self.pred_confs):
            all_low_conf = result < max_proba
            if False not in all_low_conf:
                flagged_idx.append(i)

        return flagged_idx

    def WrongPrediction(self):

        return np.where(self.labels != self.preds)[0].tolist()

    def ShortConfidence(self, conf=0.4):

        flagged_idx = []

        for i, correct_answer in enumerate(self.labels):
            if self.pred_confs[i][correct_answer] < conf:
                flagged_idx.append(i)

        return flagged_idx

    def LongConfidence(self, conf=0.2):

        flagged_idx = []

        for i, correct_answer in enumerate(self.labels):
            flagged = False
            for conf_level in np.delete(self.pred_confs[i], correct_answer):
                if conf_level > conf:
                    flagged = True
            if flagged:
                flagged_idx.append(i)
                
        return flagged_idx



    

    

    

    





            


                
