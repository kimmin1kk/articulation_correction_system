import torch

class PaddingCollate:
    @staticmethod
    def __call__(batch):
        time = [item[0] for item in batch]  # 시간 데이터
        pin1 = [item[1] for item in batch]   # pin1 데이터
        pin2 = [item[2] for item in batch]   # pin2 데이터

        # 각 시퀀스의 최대 길이를 계산합니다.
        max_length = max(len(seq) for seq in pin1 + pin2)

        # 패딩된 시퀀스를 저장할 리스트를 초기화합니다.
        padded_pin1 = []
        padded_pin2 = []

        # 배치 내의 각 시퀀스에 대해 패딩을 수행합니다.
        for seq1, seq2 in zip(pin1, pin2):
            padded_pin1.append(PaddingCollate.pad_sequence(seq1, max_length))
            padded_pin2.append(PaddingCollate.pad_sequence(seq2, max_length))

        # 리스트를 텐서로 변환합니다.
        time_tensor = torch.stack(time)
        padded_pin1_tensor = torch.stack(padded_pin1)
        padded_pin2_tensor = torch.stack(padded_pin2)

        # 패딩된 데이터와 시간 데이터를 반환합니다.
        return time_tensor, padded_pin1_tensor, padded_pin2_tensor

    @staticmethod
    def pad_sequence(sequence, max_length):
        padding_length = max_length - sequence.size(0)  # 부족한 부분의 길이 계산
        padded_sequence = torch.cat([sequence, torch.zeros(padding_length, dtype=torch.int64)], dim=0)  # 패딩된 텐서 생성
        return padded_sequence
