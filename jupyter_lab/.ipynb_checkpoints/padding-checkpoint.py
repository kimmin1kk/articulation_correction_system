import torch

class PaddingCollate:
    @staticmethod
    def __call__(batch):
        time = [item[0] for item in batch]
        pin1 = [item[1] for item in batch]
        pin2 = [item[2] for item in batch]

        max_length = max(len(seq) for seq in pin1 + pin2)

        padded_pin1 = []
        padded_pin2 = []

        for seq1, seq2 in zip(pin1, pin2):
            # 리스트를 텐서로 변환하여 패딩 수행
            padded_pin1.append(PaddingCollate.pad_sequence(torch.tensor(seq1), max_length))
            padded_pin2.append(PaddingCollate.pad_sequence(torch.tensor(seq2), max_length))

        time_tensor = torch.stack(time)
        padded_pin1_tensor = torch.stack(padded_pin1)
        padded_pin2_tensor = torch.stack(padded_pin2)

        return time_tensor, padded_pin1_tensor, padded_pin2_tensor

    @staticmethod
    def pad_sequence(sequence, max_length):
        padding_length = max_length - sequence.size(0)
        padded_sequence = torch.nn.functional.pad(sequence, (0, padding_length))
        return padded_sequence