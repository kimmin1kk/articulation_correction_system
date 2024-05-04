import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 초기 hidden state 및 cell state 생성
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 입력 텐서에 차원을 추가하여 3D 텐서로 변환
        x = x.to(torch.float32)  # 입력 데이터를 float32로 변환
        x = x.unsqueeze(1)  # 차원을 추가하여 (batch_size, 1, input_size)로 변환
        
        # LSTM 모델의 forward pass 수행
        out, _ = self.lstm(x, (h0, c0))
        
        # LSTM의 마지막 시퀀스에서의 출력을 사용하여 선형 레이어를 통해 예측 수행
        out = self.fc(out[:, -1, :])
        return out