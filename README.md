# 2025-W-Research-T02-online-action-detection

```
.
├── configs/            # 설정 파일 저장소
│   └── sample.ini        # 샘플 설정 파일
├── datasets/           # 데이터 구현 저장소
│   └── oad.py          # OAD dataset
├── models/             # 모델 구현 저장소
│   └── gatv2.py        # GATv2 model
├── utils/              
│   ├── losses.py       # Loss 생성 빌더
│   ├── optim.py        # Optimizer 및 Scheduler 생성 빌더
│   ├── metrics.py      # 성능 지표 계산 로직
│   ├── logger.py       # 로그 기록 및 파일 저장 시스템 생성 빌더
│   ├── paths.py        # 실험 결과 저장을 위한 경로 관리
│   ├── trainer.py      # 학습 루프 담당 클래스
│   └── evaluator.py    # 평가 루프 담당 클래스
├── setup.py            # 설정 파일 변환 모듈
├── train.py            # 학습 실행 메인 스크립트
└── eval.py             # 평가 실행 메인 스크립트
```