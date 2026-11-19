# 문서를 다시 만들며 확인한 것

## 설명의 순서를 다시 잡았다

기존 핸드북에는 하드웨어, DMA(Direct Memory Access), 실행 경로 등 유용한 설명이 있었지만, 독자가 **장치의 제약 → 소프트웨어의 책임 → 작성한 코드의 실행**을 이어서 추적하기 어려웠다. 짧은 ATen·TCL 장과 상대적으로 긴 실행 경로 장 사이에 설명 밀도가 달랐고, 같은 용어가 여러 계층에 쓰였다. 이번 개편은 주제를 추가하는 동시에 책임의 경계를 다시 나누었다. [개편 전 원고](https://github.com/smallstep-zzz/furyno5-handbook/tree/c0c116c68d3266ce162e69aa3e027da1851ddce4).

새 구조에서는 하드웨어와 메모리 이동을 먼저 읽고, 드라이버·런타임·컴파일러가 각기 해결하는 문제를 배운다. 이후 TCL(Tensor Contraction Language)을 작성해 EDF를 만들고, 모델의 실행계획과 버킷을 다룬다. Torch·ATen과 FuryNo5 코드는 이 기반 위에서 설명한다. 마지막으로 fxbX, 포팅, 서빙, 여러 장치, FuryBoost 회고로 넓힌다.

## 요청한 범위와 대응 장

| 개발자가 이해할 내용 | 새 문서에서 찾을 곳 |
|---|---|
| RNGD, PE, TCP, 메모리 계층 | [01 하드웨어](01-hardware.md) |
| DMA와 주소·배치·수명 | [02 메모리](02-memory-dma.md) |
| 드라이버, 펌웨어, 런타임, 공식 LLM의 연결 | [03 스택](03-driver-runtime.md) |
| 데코레이터, 축, TCL 생성, EDF 컴파일·실행 | [04 TCL 실습](04-tcl-edf.md) |
| 실행그래프 생성 주체, 저장 효과, 오류·폴백 | [05 그래프와 폴백](05-graphs-fallback.md) |
| Shape, 버킷, 준비와 실행의 차이 | [06 버킷](06-buckets-shapes.md) |
| 최적화의 축과 우선순위, 빠른·느린 경로 | [07 최적화](07-optimization.md) |
| eISA 용어 검증, vISA, furiosa-opt와 스케줄 분석 | [08 저수준 프로그래밍](08-eisa-visa.md) |
| ATen 역사·등록·개수·종류 | [09 PyTorch](09-torch-aten.md) |
| FuryNo5 파일·함수·지원 범위·컴파일·재사용 | [10 구조](10-furyno5-architecture.md), [11 컴파일](11-furyno5-compiler.md) |
| 비공식 fxbX 포맷, 공식 FXB와의 관계 | [12 fxbX](12-fxbx.md) |
| 미지원 모델 포팅·FHA 자동화 | [13 포팅 실습](13-porting-fha.md) |
| FuryAwesome의 발견·요청·카탈로그, FHA의 도구 연결과 게시 범위 | [13 생태계와 포팅](13-porting-fha.md) |
| 공식 Furiosa-LLM과 opt-vLLM 서빙의 경계 | [14 서빙](14-serving-vllm.md) |
| 여러 PE·카드·서버 최적화 | [15 여러 NPU](15-multi-npu.md) |
| FuryBoost의 CPU 대비 결과와 다음 가설 | [16 실험 회고](16-furyboost.md) |
| 비효율 진단과 측정 재현 | [17 디버깅](17-debugging.md) |
| 다른 NPU의 Torch·모델 지원 방법 | [18 NPU 생태계 비교](18-npu-ecosystems.md) |
| 공식 기준선보다 나은 결과를 연구하는 모델·실행 최적화 | [19 모델 사전 최적화와 연구](19-model-optimization.md) |

## 검토 중 바뀐 핵심 주장

| 검토할 전제 | 대조 결과 | 문서에 반영한 방식 |
|---|---|---|
| 모든 ATen을 지원한다 | FuryNo5의 목표와 실제 NPU·수치 검증 범위가 다르다. | 날짜·모수·증거 수준을 가진 지원 표 |
| ATen 호출마다 TCL을 만든다 | 준비 경로에서 컴파일하고 실행 경로에서는 준비된 객체를 호출한다. | 준비·등록·FX 치환·실행을 별도 도식으로 표시 |
| EDF가 없으면 CPU로 넘어간다 | 오류, 컴파일·캐시 경로, 명시적 CPU fallback 정책을 나눠야 한다. | 실행 경로별 판정표 |
| 그래프가 없으면 런타임이 복원한다 | 모델 입력에서 그래프를 캡처하는 것과 EDF 목록에서 모델 의미를 추론하는 것은 다르다. | 입력으로 무엇을 제공했는지에 따른 분기 |
| SDK Runnable과 opt-rt는 같은 내부 실행기다 | 같은 내부 함수로 합류한다는 공개 근거를 확인하지 못했다. | 확인 가능한 loader·메모리·호출 경계만 연결 |
| eISA가 공개 커널 API다 | 요청한 이름의 공식 풀네임·독립 API를 확인하지 못했고, 공개 문서는 vISA를 설명한다. | eISA를 지우지 않고 검증 결과와 실제 vISA 실습을 함께 기록 |
| 정적 스케줄이 빠르면 전체 요청도 빠르다 | 호스트 준비·전송·큐·sampling은 별도다. | 정적 cycles와 실측 latency 구분 |
| fxbX는 항상 Python이 EDF를 하나씩 호출한다 | 계획·SDK 조건에 따라 sequence 또는 병합된 준비 경로가 존재한다. | 패키지 계열과 실행 경로 분리 |
| vLLM 플러그인 문서만 보면 지원 범위를 알 수 있다 | 동일 커밋의 문서와 코드·테스트 사이에 DP와 sampling 설명 차이가 있었다. | 불일치 자체와 제한 조건을 기록 |

각 행의 원문과 함수 링크는 대응 장에 붙였다. 이 표는 모든 과거 문장이 틀렸다는 평가가 아니다. 질문에 포함된 전제, 기존 설명의 모호함, 버전 변화가 섞여 있었으므로 재사용할 문장을 더 엄격하게 만든 것이다.

## 테스트 결과를 읽는 기준

**문서 검사와 장치 검증은 다르다.** 이번 편집에서는 소스·CLI 정의를 대조하고, 예제 구문과 내부 링크·페이지 구성을 확인한다. 이 환경에는 Furiosa SDK와 RNGD가 없으므로 새 EDF의 생성 성공, NPU의 정확도·속도, 멀티카드 확장을 새로 실측했다고 쓰지 않는다.

FuryBoost는 저장된 실험의 설정·시각·준비 조건을 함께 읽었다. 초기 5개 트리 실험과 이후 30개 트리 실험, 첫 준비 비용과 warm 학습을 분리했다. CPU XGBoost·LightGBM이 더 빨랐다는 결과를 숨기지 않으며, 같은 트리 수와 깊이라도 알고리즘과 정확도가 다르면 완전한 동등 비교라고 부르지 않는다. [실측 기록과 해석](16-furyboost.md).

브라우저의 조절 도구는 설명을 위한 단순 모델이다. 버킷의 패딩량, 복사와 계산의 이상적 겹침, 부분 개선이 전체 시간에 미치는 효과를 보여준다. 장비의 지연·대역폭·최적 설정을 예측하는 모델로 사용하지 않는다.

## 이후 개발에서 닫아야 할 질문

| 과제 | 확인할 구현·계약 | 완료라고 판단할 증거 |
|---|---|---|
| 공식 Furiosa-LLM과 커스텀 실행계획 연결 | 모델 등록, 로더, 가중치·KV·호출 계약 | 커스텀 모델의 새 프로세스 적재와 반복 생성 |
| FuryNo5·fxbX와 vLLM 연결 | worker, batch packing, KV block, sampling, 취소·오류 | 여러 요청의 수치·상태 독립성과 end-to-end 지연 |
| 여러 장치에서 효율적인 실행 | 분할·복제, 통신, 장치별 메모리, cache locality | 한 장 대비 동일 품질의 scaling 곡선과 병목 추적 |
| FuryBoost가 경쟁력 있는 조건 | 준비 상각, 큰 histogram, 후보 축 병렬화, 정밀도 | 고정된 정확도·데이터 조건에서 CPU 기준선과 재대조 |
| 지원 연산 범위 확대 | 이름뿐 아니라 shape·dtype·stride·상태 의미 | 새 Core 목록에 대한 검증 증거와 회귀 결과 |

사용자마다 지연시간, 처리량, 컴파일 비용, 메모리, 구현 난이도의 우선순위가 다르다. 따라서 이 책은 하나의 최적화 순서를 공식 규칙처럼 고정하지 않는다. **어떤 증거로 다음 변경을 선택할지**를 공통 기준으로 둔다.

## 추가 검토: 생태계, 컴파일러, EDF와 경로 그림

사용자의 후속 질문에 따라 기존 장을 보강하고 비교·연구 장을 추가했다. 이번에도 문서 작성 환경에서 새 NPU 실행이나 성능 우위를 입증한 것은 아니다.

| 후속 질문 | 반영한 내용 | 확인 수준 |
|---|---|---|
| FHA는 FuryAwesome·FuryNo5·fxbX를 어떻게 사용하는가? | 발견·요청 → 진단·변환 → 실행계획·패키지 → 검증 → 승인된 공유의 관계도와 실제 함수·명령 | 고정 소스 확인. 카탈로그 수용범위와 현재 FHA 자동 게시기의 legacy 포맷 범위는 다름 |
| 컴파일러가 내부적으로 하는 일은? | shape·dtype·축, IR·lowering, fusion·tile·mapping, 메모리 수명, DMA·계산 스케줄, 코드 생성 | 공식 설명과 공개 검증 함수 확인. 비공개 pass의 구체적인 선택 알고리즘은 추정하지 않음 |
| EDF는 바이너리인가? | Executable Device Format이라는 이름과 실행 바이너리 성격, 역직렬화·장치 적재·버퍼 연결의 차이 | 공식 Kernel Validation과 SDK 소개, 실제 loader 대조 |
| vISA에서 runtime으로 바로 가는 그림이 맞는가? | 컴파일된 실행물과 Device Function Image를 추가하고 적재·실행 경계를 표시 | vISA는 중간 표현. SDK EDF와 opt Image의 포맷·ABI 동일성은 별도 미확인 |
| 다른 NPU는 Torch를 어떻게 지원하는가? | Ascend, Rebellions, Neuron, Tenstorrent의 연산·그래프·모델 진입점 비교 | 해당 공식 경로의 문서·소스 비교. 업체별 전체 지원·성능 순위로 일반화하지 않음 |
| 모델을 미리 바꿔 더 좋은 결과를 낼 수 있는가? | NetsPresso의 공개 범위와 오픈소스 도구, 의미 보존·품질 허용 변환, 하드웨어에 맞춘 연구 후보 | 라이브러리·논문·API 확인. RNGD에서의 이득은 후속 실측 과제 |

처음 개편한 그림은 실행기 구분에 집중하면서 컴파일 산출물을 생략했다. 그 결과 Rust·vISA를 `furiosa-opt-rt`가 직접 해석하는 것처럼 읽힐 수 있었다. [홈의 새 그림](00-reading-map.md)은 컴파일·산출물·적재·실행을 분리하고, [08장](08-eisa-visa.md)은 공개 Image·Executable·Registry 구조와 실제 코드 적재·인수 연결을 설명한다. 같은 EDF 명칭을 사용하는 것에서 교차 로딩 성공을 추론하지 않는다.

추가로 EDF 풀네임을 확인해 기존 “미확인” 표기를 수정했다. [공식 Kernel Validation의 고정 원문](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/quick-start/kernel-validation.md)은 Executable Device Format을 명시한다. 검토 결과가 바뀌면 이전의 불확실성도 근거와 함께 갱신한다.

## 문서를 유지하는 방법

코드 커밋이 바뀌면 함수 존재 여부와 성공 조건부터 다시 확인한다. 버전이 바뀌면 설치 명령·CLI 옵션·직렬화 호환성을 확인한다. 성능 결과가 바뀌면 숫자뿐 아니라 입력·가중치·threads·준비·cache 조건을 함께 교체한다. 공개 문서와 구현이 어긋난 경우에는 둘 중 하나를 조용히 숨기지 않고 조회 시점과 차이를 기록한다.

원고는 `content/`, 사이트 생성기는 `tools/build.mjs`, 전체 인용 URL 목록은 [source-index.json](data/source-index.json)에 있다. 생성된 HTML과 SVG는 외부 CDN 없이 읽을 수 있다. 내용 수정 후 `npm run build`와 `npm run check`로 문서 구조를 다시 확인한다. 그림 구조를 바꾸어 새 SVG를 생성할 때는 Graphviz와 한국어 글꼴이 필요하다.

## 출처

| 자료 | 용도 |
|---|---|
| [개편 전 핸드북](https://github.com/smallstep-zzz/furyno5-handbook/tree/c0c116c68d3266ce162e69aa3e027da1851ddce4) | 구조·원고 비교 |
| [FuryNo5 검증 범위](10-furyno5-architecture.md) | 목표와 구현 구분, 원본 소스 링크 |
| [fxbX 실행 경로](12-fxbx.md) | 최신 실행기별 조건 |
| [vLLM 코드 대조](14-serving-vllm.md) | 문서·코드 불일치와 통합 과제 |
| [FuryBoost 실측](16-furyboost.md) | CPU 비교와 개선 기록 |
