5/27

randaugment + cutmix + labelsmoothing 

randaugment + cutmix + labelsmoothing + LearningRate scheduler 

비교결과 스케줄러가 있을경우 더 빠르게 수렴이 가능하며 후반부 fine tunning까지 가능하여 정확도가 1%증가
그러나 best model이 loss가 낮은지점을 찾았으나 정확도가 낮다 -> loss가 sharp minimum에 진입하여 낮다고 생각
1. best model을 2개를 저장하여 best loss , best acc로 저장
2. SAM optimizer 를 기존 adam optimizer 또는 SGD + Momentum optimizer 에 적용하여 flat minimum으로 수렴하도록 유도
![image](https://github.com/user-attachments/assets/15411cba-489d-4070-983d-b7c212bfe8ec)
