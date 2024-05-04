import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list  # 데이터는 딕셔너리의 리스트입니다.
        
        # 최대 길이를 구합니다.
        self.max_length = max(len(data['pin1']) for data in data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]  # 인덱스를 기반으로 데이터 포인트에 접근합니다.
        
        # 데이터를 텐서로 변환하고 필요한 전처리를 수행합니다.
        x = torch.tensor(data_point['time'], dtype=torch.float)
        
        # pin1과 pin2를 패딩하여 텐서로 변환합니다.
        y = self.pad_sequence(data_point['pin1'], self.max_length)
        z = self.pad_sequence(data_point['pin2'], self.max_length)
        
        return x, y, z
    
    def pad_sequence(self, sequence, max_length):
        padded_sequence = sequence + [0] * (max_length - len(sequence))
        return torch.tensor(padded_sequence, dtype=torch.int64)
