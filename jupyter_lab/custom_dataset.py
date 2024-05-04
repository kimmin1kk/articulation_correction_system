import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list  # 이제 데이터는 딕셔너리의 리스트입니다.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]  # 인덱스를 기반으로 데이터 포인트에 접근합니다.
        # 데이터를 텐서로 변환하고 필요한 전처리를 수행
        x = torch.tensor(data_point['time'], dtype=torch.float)
        y = torch.tensor(data_point['pin1'], dtype=torch.int64)  # 리스트 형태의 데이터도 텐서로 변환 가능합니다.
        z = torch.tensor(data_point['pin2'], dtype=torch.int64)
        return x, y, z