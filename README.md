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


## 250525_1040
### main_aug_hy_randaugment
- randaug + cutmix 적용 검토 완료, 완전히 수렴하진 않아 추가 검토 필요

### main_torch_EffNet
- 작동은 하나 기존 main_torch_aug 보다 성능이 떨어짐. 추가 검토 필요

### main_torch_100
- MTL 기법으로 동작 확인

### main_100_aug
- L2 Norm 적용


## 250603_1210
### main_torch_100_aug
- MTL 기법 및 ResNet 구조 테스트

### main_100_ResNet
- 파이토치 성능 확인 및 파이토치 구조로 변경
- 모델 구조 변경에 따른 저장 방식 변경 (npz -> pkl), 피클 파일이 세이브로드가 훨씬 간단함(작동 확인 완료)
- fine test set 기준 성능 약 55%, 현재 레이어 클래스 및 트레이너는 제미나이로 작성했으므로, 기존 코드와 비교하며 세밀한 검토 필요. 얘가 L2 Norm도 적용해줬음
- Cupy 적용 검토 필요

### common.layers
- ResNet용 BatchNorm2d, AveragePooling 추가


## TODO
- EfficientNet 적용 검토 : MTL이나 ResNet보다 잘 나오는진 모르겠음
- Cupy를 활용한 numpy gpu 연산 적용 필요