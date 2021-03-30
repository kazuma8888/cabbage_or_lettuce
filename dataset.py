class TestDataset():
    def __init__(self, img, transform= None):
        self.transform = transform
        self.img = img    
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        x = self.img
        if self.transform:
            x = self.transform(x)
        return x