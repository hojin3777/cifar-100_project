# cifar-100_project
CIFAR-100 Classification Without Deeplearning Libraries

## 250518_1130
### main_
- main에서 모델 save 및 load 안정적 동작 확인

### main_aug
- main_aug 증강 클래스 및 CustomTrain 수정 완료(동작 확인 및 검증 필요)
- main_ 참조하여 save&load 함수 수정 필요

### 공통
- batchnorm=False일때 수렴하지 않는 이슈 확인 필요
- L2 Norm 적용 검토
