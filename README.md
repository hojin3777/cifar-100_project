# cifar-100_project
CIFAR-100 Classification Without Deeplearning Libraries

## 250518_1130
### main_
- main에서 모델 save 및 load 안정적 동작 확인

### main_aug
- main_aug 증강 클래스 및 CustomTrain 수정 완료(동작 확인 및 검증 필요) - 동작 확인
- main_ 참조하여 save&load 함수 수정 필요 - 완료

### 공통
- batchnorm=False일때 수렴하지 않는 이슈 확인 필요 - he initialization 으로 해결
- L2 Norm 적용 검토

## 250525_1040

### main_aug_hy_randaugment
- randaug + cutmix 적용 검토 완료, 완전히 수렴하진 않아 추가 검토 필요

### main_torch_EffNet
- 작동은 하나 기존 main_torch_aug 보다 성능이 떨어짐. 추가 검토 필요

### main_torch_100
- MTL 기법으로 동작 확인

### main_100_aug
- L2 Norm 적용


## TODO
- EfficientNet 적용 검토