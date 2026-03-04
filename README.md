# 2025-W-Research-T02-online-action-detection

## 디렉토리 구조
```
.
├── configs/            # 설정 파일 저장소
│   └── sample.ini      # 샘플 설정 파일
├── datasets/           # 데이터 구현 저장소
│   ├── oad.py          # OAD dataset
│   └── pkummd.py       # PKUMMD dataset
├── models/             # 모델 구현 저장소
│   └── gatv2.py        # GATv2 model
├── utils/              
│   ├── losses.py       # Loss 생성 빌더
│   ├── optim.py        # Optimizer 및 Scheduler 생성 빌더
│   ├── logger.py       # 로그 기록 및 파일 저장 시스템 생성 빌더
│   ├── env.py          # 실험 설정 관리 모듈
│   ├── metrics.py      # 성능 지표 계산 로직
│   ├── trainer.py      # 학습 루프 담당 클래스
│   └── evaluator.py    # 평가 루프 담당 클래스
├── train.py            # 학습 실행 메인 스크립트
└── eval.py             # 평가 실행 메인 스크립트
```

## 모델 학습

```
python train.py --config 설정파일경로(예:configs/sample.ini)
```

- 실험 결과는 `work_dirs` 경로에 저장됩니다.

## 모델 평가

```
python eval.py --config 설정파일경로(예:configs/sample.ini)
```

- 설정 파일에 적힌 `exp_dir` 경로에 있는 모델을 평가합니다.