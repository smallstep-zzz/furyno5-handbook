# 공식 SDK로 컴파일·실행·서빙하기

검증일: 2026-11-23(KST). 검증 장비: RNGD 1대. 장치 번호는 실행 환경에 맞게 선택한다.
SDK: furiosa-torch/tcc/llm 2026.4.0, Torch 2.10.0+cpu, Python 3.11.
FuryNo5·fxbX·FHA를 import하거나 실행하지 않았다. `PYTHONPATH`를 해제하고 기존 공식 SDK 환경만 사용했다.

## 준비

아래 명령은 호환 SDK·드라이버가 설치된 Linux RNGD 서버에서 실행한다. 기존 SDK Python 환경을 먼저 활성화한다. SDK·드라이버 설치 절차는 [3장](03-driver-runtime.md)을 참고한다.

핸드북 소스에 재현용 샘플을 함께 보관했다. 처음 한 번 내려받는다.

```bash
git clone --depth 1 https://github.com/smallstep-zzz/furyno5-handbook.git
cd furyno5-handbook/examples/official-sdk
source env.sh
```

`SDK_PY`는 현재 환경의 Python이다. `python -m pip show furiosa-torch furiosa-tcc furiosa-llm torch`로 버전을 확인한다. 아래 시험 기록은 SDK 2026.4.0 기준이며, 최신 버전으로 자동 치환해 검증한 결과가 아니다. `SAMPLE_LLM_DEVICE`에는 배정받은 NPU를 지정한다.

```bash
export SAMPLE_LLM_DEVICE=npu:0
```

일반 PyTorch 샘플의 `furiosa:0`은 SDK의 논리 장치 번호이고, LLM의 `npu:0`은 LLM 장치 주소다. 같은 숫자라도 표기 체계가 다르므로 사용 가능한 장치를 먼저 확인한다.

```bash
furiosa-smi status
furiosa-smi ps
```

공식 경로는 두 가지다. 일반 그래프/직접 커널은 `furiosa.torch`의 CompileModule·TclModule·Runnable을 사용한다. 공식 등록 LLM은 `fxb build`와 `furiosa_llm.LLM` 또는 `furiosa-llm serve`를 사용한다. 임의의 그래프 실행물을 `.fxb`로 이름만 바꿔 공식 LLM 실행기로 넘길 수는 없다. 모델 자산(config/tokenizer/weights)과 공식 FXB는 별개다.

## 컴파일: 성공한 방법

### 1. 모델 파일 → 공식 PyTorch 컴파일

```bash
"$SDK_PY" sample.py make
"$SDK_PY" sample.py torch
```

`model.pt2`는 torch.export로 저장한 실제 모델 파일이다. 입력은 BF16 [1,128,64]와 [1,64,128], 출력은 [1,128,128]인 작은 bmm 모델이다. `torch`는 파일을 읽어 `CompileModule.from_module`로 컴파일하고 `torch.runnable`에 공식 직렬화 실행물을 저장한 다음 실제 NPU에서 검사한다. 가중치 없는 행렬곱 예제이며 대형 체크포인트나 임의 연산 지원을 증명하지 않는다.

### 2. 모델의 해당 연산을 TCL로 명시적 연결

```bash
"$SDK_PY" sample.py tcl
```

이 샘플 모델의 유일한 bmm를 같은 입출력 계약의 `TclModule(matmul.tc)`로 명시적으로 대체한다. 원본 model.pt2의 결과도 확인하고 공식 CompileModule로 컴파일·실행한다. 모델 파일 옆에 TCL을 두면 자동 매칭된다는 뜻이 아니다. 일반 모델은 적용할 연산과 forward 연결을 직접 작성해야 한다.

고정 크기 TCL만 EDF로 컴파일하는 별도 공식 CLI도 통과했다:

```bash
furiosa-tcc matmul-fixed.tc --target-npu renegade-8pe --output matmul.edf
```

CLI 컴파일 35.1초. 기호 축(B/M/K/N)이 남은 matmul.tc를 그대로 주면 shape 변환 단계에서 실패했다. make는 동일한 커널의 축을 실제 크기로 특수화해 matmul-fixed.tc도 만든다.

주의: 이 CLI EDF를 `from_serialized_edf`로 직접 로드한 추가 시험은 `kernel size per cluster ... must equal chunk_size * num_chunks` 경고 후 완료되지 않아 시험 프로세스를 종료했다. 따라서 **EDF 컴파일 성공만 확인했으며 이 EDF의 직접 실행은 승인하지 않는다.** 위 TclModule 경로의 실행과 torch.runnable 재적재는 별도 경로로 성공했다. EDF, 직렬화 Runnable, 공식 LLM FXB는 서로 다른 형식이다.

### 3. 공식 등록 LLM → FXB

실제로 통과한 최소 프로필:

```bash
fxb build Qwen/Qwen3-8B-FP8 ./qwen-small.fxb \
  --max-model-len 128 --tokenwise-buckets 1,128 \
  --attention-buckets '(1,128,128),(1,128,1)' \
  --no-enable-sampler --concurrency 2 --build-report --dump-tc ./official-tcl
fxb show ./qwen-small.fxb
fxb inspect ./qwen-small.fxb
```

O3(기본값), 커널 8개 모두 성공, 약 74초, 결과 약 3.6MiB. 마지막 생성 완료 로그까지 확인했다. 제한된 테스트용이며 chunked prefill·prefix caching용 extend bucket은 없다. 생성 도중 KV 상태는 사용하지만 이 최소 FXB는 128토큰 밖의 작업에 사용하지 않는다.

일반 배포는 모델의 공식 기본 bucket을 유지해 빌드한다. 다음 기본 프로필 전체 빌드는 이번에 새로 시험하지 않았다:

```bash
fxb build Qwen/Qwen3-8B-FP8 ./qwen-default.fxb --concurrency 2 --build-report
```

## 컴파일: 현재 안 되는 방법

공식 `fxb build`에는 모델과 함께 임의 TCL/vISA 파일을 주는 옵션이 없다. 다음 세 명령은 실제 실행 시 모두 인자 오류(exit 2)였다:

```bash
fxb build Qwen/Qwen3-8B-FP8 invalid.fxb --tcl matmul.tc
fxb build Qwen/Qwen3-8B-FP8 invalid.fxb --visa visa
fxb build Qwen/Qwen3-8B-FP8 invalid.fxb --tcl matmul.tc --visa visa
```

직접 vISA 커널을 컴파일하는 공식 Python API는 존재한다:

```bash
"$SDK_PY" sample.py visa
# 내부 호출: furiosa.torch.compile_from_visa('./visa', 'identity')
```

하지만 검증 환경의 cargo-furiosa-opt는 0.10.0, SDK 2026.4.0 Python 래퍼 허용 범위는 >=0.5.0,<0.6.0이다. 실제 결과는 `Incompatible furiosa-opt version 0.10.0; expected it to be compatible with 0.5.0`였다. 버전 검사를 강제로 건너뛰지 않았고 전역 툴체인을 변경하지 않았다. visa/ 소스는 검사 전에 차단되어 소스 컴파일·수치 정확성도 미검증이다.

모델+vISA, 모델+TCL+vISA는 호환 툴체인을 별도 준비하고 각 커널의 ABI/forward 연결을 직접 작성한 뒤 검증해야 한다. 이 환경에서 조합 완료·실행 성공으로 주장하지 않는다. TCL과 vISA가 무조건 하나의 컴파일에 함께 들어갈 수 없다는 의미도 아니다.

## 실행: 성공한 명령

### 공식 FXB + 원본 HF 모델 → Furiosa-LLM Python

```bash
"$SDK_PY" llm_generate.py
```

실제 코드는 `LLM('Qwen/Qwen3-8B-FP8',fxb='./qwen-small.fxb',devices=장치주소,max_model_len=128)`와 `SamplingParams(temperature=0,max_tokens=8)`를 사용한다. 요청 `The capital of France is`에 ` Paris. The capital of Italy is Rome`을 생성했다. FXB만으로 실행하는 범용 `fxb run` 명령은 없고, 원본 가중치·config·tokenizer와 실행 엔진이 필요하다. LLM 장치 주소에는 `npu:번호` 표기를 사용한다. 공개 예제는 `SAMPLE_LLM_DEVICE`를 읽는다.

### PyTorch backend

```bash
"$SDK_PY" sample.py backend
```

실제 내부는 `torch.compile(model,backend=furiosa.torch.backend,fullgraph=True)`다. `backend_eager_fallback`을 사용하지 않았다.

### 모델 연산 + TCL

```bash
"$SDK_PY" sample.py tcl
```

### 저장한 공식 Runnable 재적재

```bash
"$SDK_PY" sample.py reload
```

공식 `Runnable.serialize/deserialize`로 읽으며 FXB/EDF로 이름을 바꾸지 않는다. 이 예제에는 가중치가 없다. 가중치 있는 모델은 별도 원본 가중치 공급과 입출력 복원이 필요하다.

## 서빙

### 일반 PyTorch / TCL / 재적재 실행물

다음 중 하나를 첫 번째 터미널에서 실행한다. 작은 bmm 고정 입력을 받는 **샘플 HTTP 래퍼**이며 Furiosa가 제공하는 범용 모델 서버 CLI가 아니다.

```bash
"$SDK_PY" serve.py backend --port 18085
# 또는
"$SDK_PY" serve.py torch --port 18085
# 또는
"$SDK_PY" serve.py tcl --port 18085
# 또는
"$SDK_PY" serve.py reload --port 18085
```

다른 터미널에서:

```bash
curl -sS http://127.0.0.1:18085/predict \
  -H 'Content-Type: application/json' -d '{"sample":true}'
```

네 방식 모두 HTTP 200, 출력 shape [1,128,128], CPU 최대 절대오차 0을 확인했다. 샘플은 동시에 하나의 요청을 처리하고 호출을 실제로 NPU로 보내며 결과 이미지를 캐싱하지 않는다. 실제 모델 서비스에는 별도 입력 검증·전후처리·동시성 설계가 필요하다.

### 공식 LLM 서버

첫 번째 터미널:

```bash
furiosa-llm serve Qwen/Qwen3-8B-FP8 \
  --fxb ./qwen-small.fxb --devices "$SAMPLE_LLM_DEVICE" --max-model-len 128 \
  --host 127.0.0.1 --port 18090 --served-model-name qwen-test
```

다른 터미널:

```bash
curl -sS http://127.0.0.1:18090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-test","messages":[{"role":"user","content":"What is 2 + 2? Answer briefly."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

실제 HTTP 200, 응답 `4.`. 호출 자동 재현은 `python3 test_llm_http.py`; 일반 4경로 자동 재현은 `python3 test_http.py`. 각 자동 시험은 자기 서버만 종료한다.

내 컴퓨터에서 원격 서버에 접속하려면 로컬 터미널에서:

```bash
ssh -N -L 18090:127.0.0.1:18090 USER@NPU_SERVER
```

이후 내 컴퓨터에서도 같은 curl URL을 쓴다. 터널링 명령 자체는 이번 시험에서 실행하지 않았다.

## 증거와 한계

- sample 모드(torch/tcl/reload/backend)와 각 HTTP 요청에서 공식 trace `::Task` NPU 이벤트 1개씩, DfgExecutor 이벤트 0개. 공식 SDK CPU fallback 설정 false, CPU 기준 atol=rtol=0.05; 관찰된 최대 절대오차는 0.
- LLM은 공식 native engine의 NPU 적재·KV cache·실제 생성과 HTTP 응답을 확인했다. LLM의 연산별 profiler trace/수치 기준 비교는 이번 작업 범위에서 수행하지 않았다.
- 샘플 elapsed_seconds는 import 제외, 컴파일/적재/실행을 포함하므로 순수 NPU 지연시간으로 비교하지 않는다.
- 이번 테스트 서버는 모두 종료했고 마지막 furiosa-smi ps는 비어 있었다. 원래 서비스·환경·SDK 버전은 변경하지 않았다.
- 원시 trace·JSON·로그는 실험 환경에 보존했으며 공개 예제에는 포함하지 않는다. 명령을 다시 실행하면 현재 폴더에 새 결과가 생성된다. 공개 예제는 Python 경로와 장치 선택을 환경변수로 바꾼 이식본이며, 이 문서 편집 중 NPU 실행을 반복하지 않았다.

## 출처와 검증 기준

[공식 FXB 문서](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html)

[공식 서버 문서](https://developer.furiosa.ai/stable/en/furiosa_llm/furiosa-llm-serve.html)
설치된 공식 TCL 예제: `python -m furiosa.tcl.examples.contract --target furiosa`


설치된 SDK의 `furiosa/torch/_module/{dfg,runnable,visa}.py`, `furiosa/torch/_C/{ir,profiler}/__init__.pyi`, `furiosa/tcl/examples/contract.py`와 실제 CLI 도움말을 대조했다. `compile_from_visa`가 조사 당시 미확인이었던 [8장](08-eisa-visa.md)과 달리, 이번 설치본에는 API가 있지만 버전 검사에서 차단됨을 확인했다. 공식 API·직렬화 형식·공개 opt image를 같은 형식으로 취급하지 않는다.
