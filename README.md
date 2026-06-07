# JAM Transformer — 멜로디 조건부 반주 생성

멜로디(MIDI)를 입력하면 어울리는 **반주(accompaniment)**를 생성하는 디코더-온리 Transformer
(약 37.9M 파라미터). REMI 계열 토크나이저 + **bar-block 인터리빙** 포맷 + **키-상대 화성 인코딩**을
사용하며, POP909 / Slakh / Lakh 데이터로 학습했습니다.

> 자세한 방법·진단·결과는 **[`report/report.md`](report/report.md)** 참고.

## 핵심 기여 (요약)
- 생성물이 **모든 코드를 1박에 몰아치는 "beat-1 collapse"** 문제를 발견.
- **val_loss·teacher-forced 지표가 이 문제에 눈이 먼다**는 것을 진단하고,
  모델의 *자기회귀 생성 결과*를 직접 측정하는 진단 도구
  (`scripts/analysis/generation_rhythm_stats.py`)를 구현.
- **데이터 재구성(POP909 중심) + 스케줄드 샘플링(노출 편향 교정)**으로
  리듬을 복원하면서 화성은 보존 → `back_half_share` 0.11 → 0.40 (GT 0.51),
  화성 다양성(chroma_entropy) ~1.9 유지.

## 설치
```bash
pip install -e ".[train,dev]"          # 학습/테스트
pip install -e ".[train,dev,render,demo]"  # 데모(app.py) + WAV 렌더까지
```
Python 3.11, PyTorch 2.4 (CUDA) 기준. WAV 렌더링은 `pyfluidsynth` + 사운드폰트가 필요합니다.

## 데모 (가장 빠르게 확인)
```bash
python app.py --checkpoint "checkpoints/best-epoch=007-val_loss=0.8431.ckpt"
```
브라우저에서 멜로디 MIDI를 넣으면 반주를 생성하고 입력/반주/믹스 WAV를 들려줍니다.
> 제출 모델 2개: **ep7**(리듬 분산이 가장 GT스러움, 권장)과 **ep15**(화음이 얇고 화성이 풍부).
> 각 ~434MB라 GitHub repo에는 넣지 않고 **[GitHub Release](../../releases/tag/v1.0)**에 올렸습니다.
> 다운로드 후 `checkpoints/`에 두세요:
> ```bash
> gh release download v1.0 --repo stemkwk/melody-to-accompaniment-transformer --dir checkpoints
> ```

## 재현 가능한 추론 (Docker 권장)
다른 컴퓨터에서도 동일하게 동작하도록 **CPU 전용 추론 이미지**를 제공합니다 (GPU/CUDA 불필요).
torch 버전·`fluidsynth` 시스템 라이브러리·사운드폰트를 모두 이미지에 고정해, venv로는 깨지기 쉬운
의존성을 제거했습니다.
```bash
# 1) 체크포인트 받기 (Release) → checkpoints/
gh release download v1.0 --repo stemkwk/melody-to-accompaniment-transformer --dir checkpoints
# 2) 빌드 + 실행
docker compose up --build          # 또는: docker build -t jam-infer . && docker run --rm -p 7860:7860 -v "$PWD/checkpoints:/app/checkpoints" jam-infer
# 3) 브라우저에서 http://localhost:7860
```
**venv 대안(가벼움).** Docker가 없으면:
```bash
python -m venv .venv && . .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-inference.txt && pip install -e . --no-deps
python app.py --checkpoint "checkpoints/best-epoch=007-val_loss=0.8431.ckpt"
```
> ⚠️ venv에서 **WAV 렌더링**은 시스템 `fluidsynth` + GM 사운드폰트가 필요합니다
> (Linux `sudo apt install fluidsynth fluid-soundfont-gm`, macOS `brew install fluid-synth`).
> 없어도 **MIDI 생성은 됩니다** (WAV만 생략). Docker는 이 의존성이 내장되어 있습니다.

## 학습
```bash
python scripts/train.py --epochs 40 --init_weights <seed.ckpt>
# 데이터 가중치·스케줄드 샘플링·체크포인트 정책은 configs/config.yaml 참조
```
- `source_weight_*` : 코퍼스별 샘플링 가중치 (제출 모델은 POP909 60 / Slakh 40 / Lakh 0).
- `scheduled_sampling_*` : 노출 편향 교정 (max_prob 0.25, warmup 6ep).
- `--init_weights` : 워밍스타트(가중치만 로드, 옵티마이저는 새로).

## 진단 (모델 평가의 핵심)
`val_loss`가 아니라 **자기회귀 생성 통계**로 평가합니다.
```bash
python scripts/analysis/generation_rhythm_stats.py \
    --checkpoint <ckpt> --pop909 <POP909_dir> -n 20 --max_bars 16
```
출력: 생성물의 `pos0_share / pos_entropy / back_half_share / stack_rate / chroma_entropy`를
GT와 비교. `back_half_share`↑(바 전체 활용)·`chroma_entropy` 유지(화성 보존)가 좋은 모델의 지표.

## 샘플
`samples/<곡>/` 에 곡별 WAV:
- `01_melody.wav` (입력) · `02_ground_truth.wav` (원곡 반주) · `03_generated_ep7.wav` (생성, epoch 7)
> ⚠️ POP909 멜로디가 4~8마디에서 시작해 **앞부분 ~10초는 무음**입니다(정상). 그 뒤부터 들으세요.

## 저장소 구조
```
src/jam_transformer/   핵심 패키지 (model, tokenizer, dataset, lightning_module, pipeline, utils, preprocessing)
scripts/train.py       학습 진입점
scripts/analysis/      자기회귀 진단 도구
configs/config.yaml    모든 하이퍼파라미터
app.py                 Gradio 데모
checkpoints/           제출 모델 ep7·ep15 (ZIP 전용, GitHub 제외)
samples/               생성 예시 WAV
report/report.md       보고서
```

## 데이터
학습 데이터(POP909/Slakh/Lakh 토큰화 산출물)와 체크포인트는 용량 때문에 제외했습니다.
전처리 코드는 `src/jam_transformer/preprocessing/`에 포함되어 있습니다.
