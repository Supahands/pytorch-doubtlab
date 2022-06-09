from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    
    def __init__(self, image_paths, classes, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.classes = classes
        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            label = image_filepath.split("\\")[-2]
        except IndexError:
            label = image_filepath.split("/")[-2]
            
        label = self.class_to_idx[label]
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
            
        return image, label