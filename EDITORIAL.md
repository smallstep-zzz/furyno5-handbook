# 읽기 흐름과 기술적 정확성을 위한 편집 기준

이 파일은 원고를 유지하는 사람을 위한 자료다. 학습서 본문에는 편집 과정이나 AI 문체에 대한 해설을 넣지 않는다.

## 조사에서 확인한 것과 적용 범위

- Reinhart 등의 [Do LLMs write like humans?](https://arxiv.org/abs/2410.16107)는 영어의 여러 장르와 당시 GPT-4o·Llama 3 계열을 비교했다. 일부 instruction-tuned 모델은 명사화 등 특정 문법 구조를 더 많이 사용했고, 장르에 맞는 변주에도 차이를 보였다. 한국어의 모든 AI 글이나 현재 모든 모델이 항상 그렇다는 증거는 아니다.
- Sourati 등의 [The Shrinking Landscape of Linguistic Diversity](https://arxiv.org/abs/2502.11266)는 관찰 자료와 재작성 실험에서 문체 다양성이 줄어드는 현상을 다룬다. 문체가 비슷해진다는 결과와 독자가 반드시 피로해진다는 주장은 구분한다.
- McNamara 등의 [Are Good Texts Always Better?](https://doi.org/10.1207/s1532690xci1401_1)는 사전 지식과 텍스트 연결성의 상호작용을 조사했다. 생략된 관계를 독자가 스스로 메워야 하는 부담은 배경지식에 따라 달라진다. 모든 문장을 길고 자세하게 쓰라는 규칙으로 적용하지 않는다.
- Gopen·Swan의 [The Science of Scientific Writing](https://courses.ems.psu.edu/styleforstudents/print/c10_p6.html)은 독자가 이미 아는 내용에서 새 정보로 이어지는 문장 구조와 강조 위치를 설명한다. 글쓰기 원칙을 논증한 자료이며 이 학습서의 읽기 피로를 측정한 실험은 아니다.

짧은 문장, 낮은 어휘 난도, 적은 제목 수만으로 읽기 편한 글이 되지는 않는다. 이 원고에 적용하는 판단은 연구 결과를 참고한 편집 판단이며 별도 독자 실험의 결과로 표시하지 않는다.

## 이 원고에서 확인한 문제

1. 같은 정의와 주의사항을 장마다 다시 시작해 읽은 내용이 쌓이지 않았다. 앞에서 사용한 배열·모델을 이어 쓰고, 독립 조회가 필요한 용어사전과 Q&A에서 배경을 보충한다.
2. ‘연결·계약·검증·의미 보존’ 같은 명사가 행동을 대신했다. 누가 무엇을 읽고, 어떤 값을 어디에 넣으며, 그다음 무엇을 할 수 있는지 쓴다.
3. 관련 정보가 같은 비중의 표·목록으로 늘어서 있었다. 비교·주소 대응·측정값 표는 유지하고, 원인과 과정을 설명하는 표는 문장으로 풀거나 구현 참고로 옮긴다.
4. 본문이 설명을 시작할 때마다 ‘이것이 무엇을 보장하지 않는가’를 되풀이했다. 성능 가정·지원 범위·안전 조건은 필요한 곳에 남기되 일반적인 작성·검증 고지는 출처와 실습 준비 페이지에서 관리한다.
5. 강조할 단어가 너무 많아 문장의 무게가 같아졌다. 굵게 쓰기는 실행 순서에 영향을 주는 조건이나 핵심 구별에 사용한다.
6. 장을 하나씩 고치면서 문장 시작과 맺음이 비슷해졌다. 장의 역할에 따라 계산을 따라가는 설명, 실행 절차, 실험 해석, 비교 논의를 구분한다. 모든 절에 같은 틀을 강요하지 않는다.
7. 도식의 화살표가 파일 생성, 포함 관계, 실제 함수 호출을 혼용했다. 화살표의 뜻을 명시하고, 컴파일 산출물과 그것을 읽는 코드를 소스에서 대응시킨다.

## 원고를 수정할 때

문단의 첫 문장이 앞 문단에서 알게 된 사실을 이어받는지 읽는다. 새 용어가 필요하면 그 용어가 해결하는 상황부터 보여 준다. 같은 예를 되풀이하는 대신 다음 문제로 확장한다. 절 끝에서는 방금 얻은 결과를 말하고, 그 결과만으로 다음 단계에서 무엇이 남는지 연결한다.

정의의 나열을 부드러운 말투로 바꾸는 것으로 끝내지 않는다. 서술 순서, 예제의 일관성, 중요도 차이를 함께 고친다. 연결 문장을 추가할 때는 주변의 중복 요약을 덜어내어 총 설명을 불필요하게 늘리지 않는다.

코드·입력 조건·예외 처리·실측 원자료를 문체를 위해 지우지 않는다. 읽기 흐름에서 분리할 구현 표와 환경 재현 명령은 제목을 분명히 한 참고 영역에 보관한다. 근거 없는 비유, 새 API, 실제로 실행하지 않은 출력은 만들지 않는다.

## 실행 경로도 대조 결과

2026-11-21에 다음 고정 소스와 공식 FXB 문서를 대조했다. 이는 소스 확인이며 새 NPU 실행 결과가 아니다.

| 경로 | 확인한 연결 | 근거 |
|---|---|---|
| 공식 모델 | 모델 자산과 FXB를 Furiosa-LLM이 함께 읽는다. FXB만으로 가중치·tokenizer·설정이 대체되지 않는다. | [공식 FXB](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html) |
| FuryNo5 TCL export | TCC 결과 EDF와 값 연결을 실행계획으로 묶어 export한다. | [tcl_compiler.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/tcl_compiler.py), [execution_plan.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py#L1243-L1297) |
| fxbX 소비자 | 패키지 검사 → EDF 해석 → SDK Runnable 적재 → 조건에 따라 sequence 구성. 실제 입력·가중치는 별도 연결한다. | [execution_fxb.py](https://github.com/smallstep-zzz/fxbX/blob/03d993b4d1b36cdda209d61bbf447434c174b401/fxbex/execution_fxb.py#L392-L439) |
| FuryNo5 direct | `ft.compiler.compile` 결과를 SDK 형식으로 직렬화·복원하고 `ft.Runnable`로 적재한다. 이 직렬화를 EDF라고 바꾸어 부르지 않는다. | [direct_runtime.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/direct_runtime.py) |
| 공개 opt | compiler image는 장치 코드·메모리 바인딩 등을 포함한다. link pass가 registry에 기록하고 runtime이 Image를 파싱해 Function을 적재한다. | [image.rs](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-abi/src/image.rs), [registry](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/src/registry/mod.rs), [loader](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/src/registry/load/mod.rs) |

기존 그림은 다음 구별을 반영해 교체한다.

- `EDF → Device Function Image`를 별도의 변환 공정으로 확인한 화살표처럼 그리지 않는다.
- fxbX는 SDK와 나란한 언어·하드웨어 계층이 아니라 계획을 읽고 SDK를 호출하는 소비자다.
- SDK 직렬화, TCC EDF, opt image의 상호 적재 호환성을 이 대조 결과에서 추론하지 않는다.
