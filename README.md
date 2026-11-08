<div align="center">

# Robotics Roadmap

**로보틱스 개발자가 되기 위한 오픈소스 독학 커리큘럼**

[![Lessons](https://img.shields.io/badge/lessons-57-2dd4bf)](#커리큘럼)
[![Hours](https://img.shields.io/badge/hours-120-818cf8)](#커리큘럼)
[![Lessons license](https://img.shields.io/badge/lessons-CC%20BY--NC--SA%204.0-lightgrey)](lessons/LICENSE.md)
[![Site license](https://img.shields.io/badge/site-MIT-blue)](LICENSE)
[![Node](https://img.shields.io/badge/node-%E2%89%A520.19-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![GitHub stars](https://img.shields.io/github/stars/woolimi/robotics-roadmap?style=social)](https://github.com/woolimi/robotics-roadmap/stargazers)

[예시 사이트](https://woolimi.github.io/robotics-roadmap/) · [설치와 실행](#설치와-실행) · [커리큘럼](#커리큘럼) · [강의와 퀴즈](#강의와-퀴즈) · [기여](#기여) · [라이선스](#라이선스)

<img src="docs/site-tour.gif" alt="커리큘럼 지도에서 강의 원을 눌러 선수·후속 강의가 강조되고, 오른쪽 창에 강의 본문이 열리는 모습" width="960">

</div>

## 소개

로보틱스 개발자가 되기 위한 독학 커리큘럼입니다. vector·미분·확률과 ROS 2 같은 공통 기초에서 출발해 robot learning, manipulation,
mobile robot·자율주행 세 직무 방향으로 나아가는 강의 57개를 담았습니다. 리포를 fork 해 지도를 띄우면 강의를 읽고 퀴즈를 풀며 혼자서
진도를 이어 갈 수 있습니다.

## 특징

- **fork 해서 혼자 진행하는 자율형 학습**: 강의를 읽고 면접형 퀴즈를 풀면 결과가 내 fork 의 `progress/` 에 쌓이고, 다시 풀 때는
  틀린 원리만 다른 문항으로 묻습니다.
- **기술의 의존 관계가 보이는 지도**: 강의가 선수 관계로 이어져 있어서, π0 를 이해하려면 RT-2·OpenVLA, flow matching, ACT 를 먼저
  읽어야 한다는 것이 지도에서 바로 보입니다.
- **논문 기반 강의**: 강의마다 근거 논문을 밝히고 그 방법과 숫자를 따라갑니다. Kalman(1960)부터 π0·GR00T N1 까지 논문·교과서·공식
  문서 200여 편을 다룹니다.
- **숫자와 그림으로 설명**: 코드 없이 손으로 따라 할 숫자 예시와 그림으로 설명하고, 긴 유도는 접어 둔 "심화" 블록에 있습니다.
- **업데이트 알림**: 원본 리포에 새 강의가 올라오면 지도 화면에서 버튼 하나로 내 fork 에 합칩니다.

## 설치와 실행

설치하기 전에 [예시 사이트](https://woolimi.github.io/robotics-roadmap/)에서 지도와 강의를 둘러보고 퀴즈를 풀어 볼 수 있습니다. 예시
사이트의 퀴즈 결과는 새로고침하면 사라지므로, 진행 기록을 남기려면 아래처럼 내 fork 를 띄워서 씁니다.

[Node.js](https://nodejs.org/) 20.19 이상(또는 22.12 이상)과 git 이 필요합니다. 퀴즈 기록을 내 GitHub 에 남길 수 있도록 fork 해서 씁니다.

1. 이 페이지 오른쪽 위의 **Fork** 를 눌러 내 계정으로 복사합니다
2. 내 fork 를 받아 지도를 띄웁니다

   ```bash
   git clone https://github.com/<내 계정>/robotics-roadmap.git
   cd robotics-roadmap/site
   npm install
   npm run dev
   ```

3. 브라우저에서 http://localhost:5173 을 엽니다

### 퀴즈 기록

퀴즈를 제출하면 지도를 띄운 dev server 가 답안을 채점해 `progress/<강의>.json` 에 저장합니다. 이 파일은 내 컴퓨터에만 있으므로,
공부를 마칠 때 커밋해서 push 해 두면 다른 컴퓨터에서 clone 해도 진행 상태가 그대로 보입니다.

```bash
git add progress
git commit -m "Record quiz progress"
git push
```

### 강의 업데이트

강의는 계속 늘어나고 고쳐집니다. `npm run dev` 로 지도를 띄우면 dev server 가 원본 리포(woolimi/robotics-roadmap)를 확인해서, 내 fork 에
없는 변경이 있으면 화면 위쪽에 업데이트할지 묻습니다. **업데이트** 를 누르면 원본의 변경을 내 fork 에 합치고, 퀴즈 기록은 그대로
남습니다. 합친 결과는 다음에 퀴즈 기록을 push 할 때 함께 올라갑니다.

원본은 `progress/` 를 건드리지 않으므로 기록만 커밋했다면 충돌하지 않습니다. `lessons/` 나 `site/` 를 직접 고쳤다면 원본의 변경과 겹쳐
자동으로 합치지 못할 수 있는데, 그때는 터미널에서 직접 합칩니다.

```bash
git fetch upstream
git merge upstream/main
```

## 커리큘럼

![커리큘럼 지도. 가운데 동심원에 공통 기초가 놓이고, 점선으로 나뉜 세 구역으로 Robot Learning·Manipulation·Mobile Autonomy 트리가 뻗습니다](docs/map.png)

| 영역 | 다루는 것 |
| --- | --- |
| 공통 기초 | vector·행렬, 미분·gradient, 확률·Gaussian, ROS 2(node·topic·tf2), 좌표 변환, PID 제어, 카메라 geometry |
| Robot Learning | linear regression 에서 PyTorch 까지의 신경망 기초, CNN·ViT·CLIP, Transformer·GPT-2, VAE·diffusion·flow matching·one-step 생성, 강화학습(DQN·SAC·PPO·sim-to-real), imitation learning(behavior cloning·DAgger·ACT·Diffusion Policy), VLA(RT-2·OpenVLA·π0·GR00T N1), world model |
| Manipulation | forward·inverse kinematics, MuJoCo simulation, force·impedance control, ros2_control, trajectory generation, motion planning(RRT·OMPL·MoveIt 2), hand-eye calibration, pick-and-place, grasp planning |
| Mobile Autonomy(이동 로봇·자율주행) | Kalman filter, particle filter, LiDAR·visual SLAM, path planning(A*·Nav2), path tracking(pure pursuit·MPC), LiDAR perception, 자율주행 stack(Autoware) |

원 하나가 강의 하나이고, 선은 먼저 읽어 두면 좋은 강의를 잇습니다. 가운데의 Vectors & Matrices 에서 시작해 공통 기초를 읽고,
지원하려는 직무의 영역을 따라 바깥으로 나아가면 됩니다. 선수 관계는 권장 순서일 뿐이라 어떤 강의든 바로 열 수 있습니다.

## 강의와 퀴즈

![Kalman filter 강의. 오른쪽 창에 숫자로 계산하는 표와 그림이 보이고, 왼쪽 지도에서는 이 강의와 선수·후속 강의가 강조됩니다](docs/lesson.png)

강의는 학습 목표, 본문, 정리 순서로 이루어집니다. 본문은 직접 따라 계산하는 표, 그림, 수식, 절마다 답을 접어 둔 확인 문제로 짜여
있고, 긴 유도는 "심화" 블록에 접어 두어 처음 읽을 때는 건너뛸 수 있습니다.

![퀴즈 화면. 개념 질문에 4지선다로 답하는 문항이 보입니다](docs/quiz.png)

강의 끝의 퀴즈는 강의의 원리 하나마다 한 문항을 묻습니다. 제출하면 틀린 문항의 해설이 보이고, 해설마다 그 내용을 다룬 절로 가는
버튼이 있습니다. 절을 읽은 뒤 강의 아래쪽의 "퀴즈 해설로 돌아가기" 로 해설에 돌아와 다음 틀린 문항을 이어 볼 수 있고, 지난 응시의
해설도 회차별로 다시 볼 수 있습니다. 다시 풀 때는 틀린 원리만 같은 원리의 다른 문항으로 묻습니다. 틀린 문항이 남지 않으면 그 강의는
완료로 칠해지고 이어지는 강의가 밝아집니다.

## 프로젝트 구조

```text
robotics-roadmap/
├── lessons/              # 강의 57개
│   ├── branches.yaml     # 지도의 동심원·영역·branch 설정
│   └── <강의>/
│       ├── README.md     # 강의 본문 (frontmatter 에 선수 강의·시간·논문)
│       ├── quiz.yaml     # 면접형 퀴즈
│       └── figures/      # SVG 그림
├── progress/             # 내 퀴즈 기록 (fork 에서만 커밋)
├── site/                 # 지도와 강의를 보여 주는 Vite + React 앱
├── docs/                 # README 이미지, 강의 문장 패턴(korean-prose.md)
├── CONTRIBUTING.md       # 강의·퀴즈·지도 규칙
└── CLAUDE.md             # Claude Code 로 강의를 쓸 때의 작업 순서와 규칙
```

## 기여

틀린 내용, 어색한 설명, 깨진 링크는 [issue](https://github.com/woolimi/robotics-roadmap/issues) 로 알려 주시면 고치겠습니다.
사실 정정은 근거가 되는 논문·공식 문서·소스 코드 링크를 함께 남겨 주시면 가장 빠르게 반영됩니다.

강의를 고치거나 새로 쓰는 pull request 도 환영합니다. 강의 형식, 문장 규칙, 퀴즈 규칙, 지도 사이트의 구조는
[CONTRIBUTING.md](CONTRIBUTING.md) 에 있고, 문장을 다듬을 때 고치는 패턴은 [docs/korean-prose.md](docs/korean-prose.md) 에
있습니다. Claude Code 로 강의를 쓰면 리포의 [CLAUDE.md](CLAUDE.md) 를 자동으로 읽어 같은 작업 순서와 사실 확인 규칙을 따릅니다. PR 을 올리기 전에 `site/` 에서 `npm test` 로 형식 검사를 통과하는지 확인해 주세요.

## Special Thanks

- [Sunbin Kim](https://github.com/SunandBean): [논문 해설 영상](https://www.youtube.com/@paper2videoai) 제공, [paper2video](https://github.com/SunandBean/paper2video) 제작자

## 라이선스

- 강의(`lessons/`)와 문서 이미지(`docs/`): [CC BY-NC-SA 4.0](lessons/LICENSE.md). 출처를 밝히면 비영리 목적으로 자유롭게 쓰고 고칠 수
  있고, 고친 자료도 같은 조건으로 공유해야 합니다
- 지도 사이트(`site/`): [MIT](LICENSE)
- 지도의 트리 그림: 로봇팔("mechanical-arm", Lorc)과 주행 로봇("mars-curiosity", Delapouite)은 [game-icons.net](https://game-icons.net) 의
  그림으로 [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), 전자 두뇌("brain-circuit")는
  [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons) 의 그림으로 MIT 입니다
