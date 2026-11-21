# FuryNo5 개발 핸드북

RNGD의 하드웨어에서 모델 포팅·서빙까지 연결하는 한국어 개발 문서입니다.

**읽기:** https://smallstep-zzz.github.io/furyno5-handbook/

공식 자료와 FuryAwesome·FuryNo5·fxbX·Furiosa-Hug-Anything의 구현을 구분합니다. 지원 목표와 검증 결과, 정적 스케줄과 실제 측정, 공식 FXB와 비공식 확장을 섞지 않습니다.

Python의 함수·배열·반복문을 읽을 수 있으면 작은 숫자 예제부터 시작할 수 있습니다. 가중치·바이어스, 배열 배치, 컴파일과 실행은 필요한 상황을 먼저 보여 준 뒤 이름을 붙입니다. 장치 API를 쓰는 부분은 준비물과 입력·출력의 관계를 설명합니다. 공식 문서와 소스 링크는 근거와 더 깊은 구현을 확인하는 데 사용합니다.

## 구성

- 01–03: RNGD, PE, 메모리, DMA, 드라이버, 런타임
- 04–08: TCL, EDF, 실행그래프, 오류·폴백, 버킷, 최적화, vISA·furiosa-opt
- 09–11: PyTorch·ATen, FuryNo5 구조·컴파일·재사용
- 12–15: fxbX, 모델 포팅·FHA·FuryAwesome, vLLM 서빙, 여러 NPU
- 16–17: FuryBoost 기존 실측과 디버깅
- 18–19: 다른 NPU의 PyTorch 통합, NetsPresso와 공개 모델 최적화 도구, RNGD 최적화 연구
- 용어 사전, 출처와 버전, 실행 전 확인사항, 핵심 질문과 해설

## 원고와 생성

원고는 `content/*.md`, 목차는 `data/chapters.json`입니다. 코드 예제는 `examples/`, 인용 URL 목록은 `data/source-index.json`입니다. `research/`는 편집 중 참고 자료용이며 저장소와 배포에 포함하지 않습니다.

```bash
npm install --ignore-scripts --no-audit --no-fund
npm run build
npm run check
npm run serve
```

Node.js 20 이상과 Python 3.10 이상을 사용합니다. 새로운 Mermaid 도식을 SVG로 생성할 때는 Graphviz(`dot`)와 Noto Sans CJK KR 글꼴이 필요합니다. 기존 도식은 SVG를 저장하므로 독자의 브라우저에서 외부 CDN이나 Mermaid 라이브러리를 불러오지 않습니다. 지원하는 도식 문법은 간단한 `flowchart TB/LR`이며, 처리하지 못하는 문법은 빌드 오류를 발생시킵니다.

## 근거와 검증 범위

소스·문헌 기준일은 2026-11-19입니다. SDK는 2026.4 계열을 중심으로 하고 2026.5 beta 기능·furiosa-opt preview는 별도로 표시합니다. 공식 문서 URL의 버전과 실제 페이지 헤더가 다른 사례도 기록했습니다. 프로젝트 코드 링크는 가능한 경우 commit으로 고정했습니다.

FuryNo5·FHA·FuryBoost는 조사 시점에 비공개 저장소였으므로 소스 링크에 권한이 필요합니다. 사용자가 요청한 핵심 코드 발췌·실험 결과만 설명에 포함하며, 조사용 원본 소스·원시 로그·호스트별 정보는 재배포하지 않습니다. fxbX는 공식 Furiosa 포맷의 범용 확장을 목표로 한 비공식 프로젝트입니다.

정적 사이트 검사와 Python 예제 구문 검사는 NPU 검증을 대신하지 않습니다. 이 편집 환경에서 RNGD 실행이나 새로운 성능 측정을 수행하지 않았습니다. FuryBoost 숫자는 명시한 저장소의 기존 실험에서 가져왔습니다.

NumPy가 설치되어 있다면 `python examples/tensor_basics.py`와 `python examples/array_layout.py`로 행렬곱·einsum·reshape·전치·stride·메모리 공유를 직접 확인할 수 있습니다. 이 두 CPU 기초 실습은 편집 환경에서 실제로 실행했습니다.

본문의 설명용 조절 도구는 행렬곱의 행·열과 M/N/K 분할, 버킷 패딩, 이상적인 복사·계산 겹침, Amdahl의 법칙을 탐색합니다. 장비 성능 예측과 구분합니다.

EDF는 공식 Kernel Validation의 Executable Device Format 표기를 사용합니다. vISA 중간 표현, 컴파일된 실행물, Device Function Image, SDK와 opt의 적재·실행 API를 구분하며, 서로의 실행물을 교차 로딩할 수 있다고 주장하지 않습니다.

## 갱신 원칙

원고를 바꾼 뒤 생성 HTML과 관련 SVG도 함께 갱신합니다. 기존 8개 장 URL은 새 문서로 이동하도록 유지합니다. 배포 전 로컬 링크·앵커·도식·예제 구문을 검사합니다. 공개 문서와 실제 코드가 다르면 조회일과 불일치를 명시합니다.

출처의 코드·자료는 각 프로젝트의 라이선스와 권리를 따릅니다.
furiosa-opt 공식 도식을 한국어로 재구성한 부분에는 원본 링크를 붙였으며, 해당 Apache-2.0 라이선스와 NOTICE를 `assets/licenses/`에 보존했습니다.
