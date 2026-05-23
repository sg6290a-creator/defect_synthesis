# Defect Synthesis Lite

적대적 결함 합성(adversarial defect synthesis) 모델을 학습 및 테스트하기 위한
경량 Gradio 패키지입니다. 이 폴더는 GitHub 공개용으로 정리되어 있어, 대용량
데이터셋, 학습된 체크포인트, 실험 실행 결과는 포함하지 않습니다.

포함된 데이터셋은 매우 작은 합성 데모 데이터셋입니다. UI와 학습 루프가 정상적으로
동작하는지 확인하는 용도로만 적합하며, 실제 모델 품질을 측정하기에는 적합하지
않습니다.

## 구성

- `ui/app.py`: 로컬 Gradio 웹 UI.
- `ui/inference.py`: 생성기 체크포인트 추론 헬퍼.
- `train.py` 및 `src/`: 모델 및 학습 코드.
- `data/demo_dataset`: 작은 합성 A/B/mask 학습용 데이터셋.
- `data/demo_input`: 생성 테스트용 깨끗한 이미지 폴더.
- `scripts/make_demo_dataset.py`: 결정론적(deterministic) 데모 데이터셋 생성기.
- `scripts/sam_make_masks.py`: 선택 사항인 SAM 기반 마스크 전처리 스크립트.

## 설치

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

SAM 전처리를 사용하려면 지원되는 SAM 백엔드 하나를 설치하고, 모델 체크포인트는
Git에 포함되지 않도록 외부에 보관하세요:

```bash
pip install -r requirements-sam.txt
```

이미 PyTorch가 설치된 conda 환경을 사용하는 경우, 해당 환경을 활성화한 뒤
다음만 실행하면 됩니다:

```bash
pip install -r requirements.txt
```

## 웹 UI 실행

```bash
python -m ui.app
```

출력된 로컬 URL을 브라우저에서 여세요. 보통 다음 주소입니다:

```text
http://127.0.0.1:7860
```

Train 탭에는 데모 데이터셋 경로가 미리 채워져 있습니다. 다음 값으로 시작해 보세요:

- 모델 이름: `demo_toy`
- 에포크 수: `2`
- 이미지 크기: `128`
- CUDA: 설치된 PyTorch가 CUDA를 사용할 수 있을 때만 활성화

학습이 끝나면 Generate 탭으로 이동해 `refresh models`를 누른 뒤, 방금 생성된
`.pt` 모델을 선택하고 `data/demo_input`을 입력으로 샘플을 생성합니다.

## SAM 전처리

Preprocess 탭은 결함 이미지 폴더로부터 `B` 이미지와 `mask` 파일을 생성할 수
있습니다. SAM은 실제 결함 영역을 알기 위해 프롬프트가 필요하므로 bbox 라벨을
함께 제공하는 것을 권장합니다.

권장 입력 구조:

```text
raw_defects/
  defect_001.png
  defect_002.png
labels/
  defect_001.txt
  defect_002.txt
```

기본 라벨 형식은 YOLO bbox입니다:

```text
class_id x_center y_center width height
```

출력 경로:

```text
data/sam_preprocessed/train/B/
data/sam_preprocessed/train/mask/
```

Ultralytics SAM 백엔드를 사용한 CLI 예시:

```bash
python scripts/sam_make_masks.py \
  --backend ultralytics \
  --checkpoint /path/to/sam_b.pt \
  --defect-dir /path/to/raw_defects \
  --labels-dir /path/to/labels \
  --out-root data/sam_preprocessed
```

Meta SAM 백엔드를 사용한 CLI 예시:

```bash
python scripts/sam_make_masks.py \
  --backend meta \
  --model-type vit_b \
  --checkpoint /path/to/sam_vit_b.pth \
  --defect-dir /path/to/raw_defects \
  --labels-dir /path/to/labels \
  --out-root data/sam_preprocessed
```

프롬프트 없이 동작하는 `--auto` 모드도 있지만, 부트스트랩 용도로만 쓰기에 적합한
약한 방식입니다. 자동 SAM 세그먼트는 결함이 아닌 영역까지 포함할 수 있으므로,
학습 전에 마스크를 반드시 확인하세요.

## 데모 데이터 재생성

```bash
python scripts/make_demo_dataset.py --count 8 --size 128
```

## 저장소 관리(Hygiene)

생성된 학습 결과는 `ui/runs/` 아래에 기록되며 Git에서 무시(ignore)됩니다.
체크포인트(`.pt`, `.pth`)도 무시되어 대용량 바이너리가 실수로 커밋되지 않도록
보호됩니다.

## 출처(Attribution)

모델 코드는 "Adversarial Defect Synthesis for Industrial Products in Low Data
Regime" 논문 및 Apache-2.0 라이선스 원본 구현에 기반합니다. 자세한 내용은
`LICENSE`와 `NOTICE` 파일을 참고하세요.
