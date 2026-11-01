# Robotics Roadmap

로보틱스 개발자가 되기 위한 한국어 독학 커리큘럼이다. `lessons/<id>/` 에 강의 57개가 있고, `site/` 의 Vite + React 앱이 강의
frontmatter 를 읽어 선수 관계 지도를 그린다. 공부하는 법은 [README.md](README.md) 에 있다.

## 먼저 읽을 문서

- [CONTRIBUTING.md](CONTRIBUTING.md): 강의 구성, frontmatter, 본문·퀴즈 규칙, 지도 사이트 구조의 정본
- [docs/korean-prose.md](docs/korean-prose.md): 문장 검수 때 고치는 한국어 패턴
- 기준 강의: [lessons/pid-control](lessons/pid-control/README.md), [lessons/kalman-filter](lessons/kalman-filter/README.md) 의
  README·`quiz.yaml`·`figures/`

## 강의 작성 순서

1. 강의의 자리를 확인한다. `lessons/branches.yaml` 에서 branch 를 찾고, `requires` 의 강의와 이 강의를 `requires` 에 둔 강의
   (`grep -l "requires:.*<id>" lessons/*/README.md`)의 제목과 숫자를 읽어 앞뒤에서 무엇을 가르치는지 본다. 가운데 hub 강의(vector·행렬,
   미분·gradient, 확률·Gaussian, ROS 2, 좌표 변환, PID, 카메라 geometry)가 가르치는 내용은 다시 가르치지 않고 한두 문장으로 짚는다
2. 근거 자료를 직접 연다. 논문 원문(arXiv), 공식 문서, 소스 코드(GitHub)에서 방법과 숫자를 확인한다. 라이브러리의 기본값은
   소스 코드에서 확인한다
3. outline 을 먼저 쓴다. 강의 전체를 한 문장으로 요약하고, 절마다 "절 | 앞 절이 남긴 문제 | 이 절의 해결 | 대가·한계와 숫자 하나 | 근거"
   를 한 줄씩 적은 표를 만든다. 근거 칸에는 숫자마다 논문의 절·표 번호, 공식 문서·소스 파일, 직접 계산 가운데 어디서 왔는지 적는다.
   outline 은 scratchpad 에 두고 리포에 넣지 않는다
4. 본문을 쓴다. 서론 → 학습 목표 → 개념마다 한 절 → 정리 → 참고 자료 순서이고, 일반 식·유도·긴 손계산은 "심화" 블록에 접는다. 절은
   outline 대로 앞 절이 남긴 문제 → 해결 → 그림 → 숫자 예시 → 실제 로봇·논문·제품 사례 → 대가·한계 순서로 쓴다. 심화 밖 본문은
   15k ~ 22k 자, 전체는 20k ~ 28k 자 안팎이다
5. 그림을 그린다. SVG 로 `figures/` 에 두고 [PID Control 의 그림](lessons/pid-control/figures/control-loop.svg) 과 같은 어두운 카드와
   색(배경 `#0b1220`, 글자 `#e2e8f0`, 강조 `#7cb4d8`·`#7fc4a2`·`#d6aa62`·`#d88b8b`·`#a99ad6`·`#cf96b8`, 보조 `#94a3b8`)을 쓴다. 그림을 만드는
   script 는 리포에 넣지 않는다. 넣기 전에 PNG 로 렌더해(headless Chrome `--headless=new --screenshot`) 글자 겹침과 잘림을 눈으로 확인하고,
   그림 속 숫자를 본문 예시와 맞춘다
6. 퀴즈를 쓴다. [lessons/kalman-filter/quiz.yaml](lessons/kalman-filter/quiz.yaml) 과 같은 모양으로 topic 6 ~ 8개, topic 마다 variant
   3개다
7. 형식을 검사한다: `cd site && npm test`
8. 사실을 따로 검증한다. 강의를 쓴 context 는 자기 강의를 판정하지 않고, 새 subagent 가 강의 파일·outline·근거 자료 원문만 보고 검증한다.
   숫자와 "저자들은 ~ 라고 적었다" 같은 주장은 원문에서 찾고, 논문이 붙인 조건(dataset, metric, hardware, 표 번호, 상대값인지 절대값인지)이
   빠지지 않았는지 보고, 표와 예시의 계산은 다시 계산하고, 같은 강의 안에서 서로 어긋나는 문장을 찾는다. 결과는
   "줄 | 강의 서술 | 원문 근거 | 판정(ok / 조건 누락 / 오해 소지 / 틀림) | 고칠 안" 표로 받는다. subagent 는 파일을 고치지 않고, 지적을 고친 뒤
   고친 항목만 다시 검증해 모두 ok 가 될 때까지 되풀이한다
9. 문장을 따로 검수한다. 다 쓴 뒤 강의 파일과 [docs/korean-prose.md](docs/korean-prose.md) 만 놓고 문장만 고치는 단계를 한 번 더
   거친다(가능하면 새 subagent 로). 사실·숫자·표·frontmatter·`](#slug)` 로 링크된 제목은 건드리지 않는다

## 사실 확인

- 숫자·날짜·기본값·인용은 원문에서 확인한 것만 쓴다. 확인하지 못한 주장은 고치거나, 단정하지 않는 서술로 바꾸거나, 지운다
- 본문의 숫자마다 논문·문서·직접 계산 중 어디서 왔는지 밝힌다. 계산을 확인한 script 는 리포에 넣지 않는다
- 숫자뿐 아니라 논문이 밝힌 결과의 이유도 원문으로 확인한다
- 서술이 직관과 다르다는 지적을 받으면 원문을 찾아 "강의 서술 | 원문 근거 | 판정" 표로 대조한 뒤 고친다
- frontmatter `papers` 와 `## 참고 자료` 에는 직접 열어 본 자료만 싣는다
- 여러 강의에 걸쳐 나오는 사실은 다른 강의의 서술과 맞춘다. 이미 원문으로 확인해 고친 예:
  - ACT 는 실행할 때 CVAE 의 z 를 prior 평균 0 으로 고정하므로, 반반 갈린 시연에서 봉우리 하나를 고른다는 보장이 없다
    ([arXiv 2304.13705](https://arxiv.org/abs/2304.13705))
  - GR00T N1 논문 식 (1) 의 target 은 같은 논문의 섞는 식·Euler 식과 맞추면 ε − Aₜ 가 아니라 Aₜ − ε 다
    ([arXiv 2503.14734](https://arxiv.org/abs/2503.14734))
  - OpenVLA 가 action bin 을 vocabulary 에 매핑하는 방식은 단정하지 않는다

## 설명 방식

- 가정하는 지식은 중학교 산수다. 처음 나오는 용어·기호·약어는 등장한 자리에서 설명하고, "뒤에서 설명한다" 로 미루지 않는다. 학습 목표,
  표, 그림, 확인 문제, 퀴즈에서 처음 나와도 같다
- 숫자 예시를 먼저 보이고 기호는 그 뒤에 쓴다. 숫자에서 일반 식으로 넘어갈 때는 "이 예의 값 | 기호로 쓴 값" 대응 표를 먼저 두고
  식은 그 표의 마지막 줄로 적는다
- 비율("1 늘 때마다 3 는다")과 실제 변화량("2 늘었다")을 한 문장에 섞지 않는다
- 값이 바뀌는 과정을 그린 그림에는 변화 전, 변화 후, 변화량을 모두 적는다
- gradient 는 부호로 설명한다(+ 면 parameter 를 늘릴 때 loss 가 늘고, − 면 줄고, 0 이면 변화가 없다)
- 정리에는 bullet 과 다음 강의 예고만 둔다. 면접 문답 문단은 넣지 않는다
- 코드 블록을 넣지 않는다. technical term 은 영어 원어로 쓴다("pretraining", "dataset")

## 작업 범위

- 강의 하나를 고칠 때는 그 강의 폴더만 바꾼다. `lessons/branches.yaml` 이나 `site/` 를 바꾸면 `npm test` 로 지도 레이아웃 검사까지 통과시킨다
- `progress/` 는 fork 에서만 커밋하는 퀴즈 기록이다. 원본 리포에는 커밋하지 않는다
- 사이트의 튜닝값과 경로는 `site/src/config.ts` 의 `CONFIG` 한 곳에 두고, 환경변수를 새로 읽지 않는다
- 커밋 메시지는 영어로 쓴다
