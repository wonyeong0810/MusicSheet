import torch
print(torch.__version__)
print(torch.version.cuda)  # None이면 CUDA 지원 안 되는 버전
# print(torch.cuda.is_available())  # False면 GPU 사용 불가
print(torch.cuda.current_device())  # 현재 GPU 디바이스 ID
print(torch.cuda.get_device_name(0))  # GPU 디바이스 이름
print(torch.cuda.device_count())  # GPU 디바이스 개수
