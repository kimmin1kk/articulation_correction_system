import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_list, max_length=None):
        self.data = data_list
        if max_length is None:
            self.max_length = max(len(data['pin1']) for data in data_list)
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        x = torch.tensor(data_point['time'], dtype=torch.float)
        
        # 리스트를 텐서로 변환하여 패딩 수행
        y = self.pad_sequence(torch.tensor(data_point['pin1']), self.max_length)
        z = self.pad_sequence(torch.tensor(data_point['pin2']), self.max_length)
        
        return x, y, z
    
    def pad_sequence(self, sequence, max_length):
        # 패딩을 위해 torch.nn.functional.pad 사용
        padded_sequence = torch.nn.functional.pad(sequence, (0, max_length - len(sequence)))
        return padded_sequence