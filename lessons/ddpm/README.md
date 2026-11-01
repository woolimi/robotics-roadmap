---
id: ddpm
title: DDPM
branch: generative
requires: [vae, cnn]
hours: 2
summary: 정답이 갈라진 데이터에서 평균 대신 한 무리를 고르는 생성 방법. 데이터를 noise 로 망가뜨리는 forward process, 섞인 noise 를 맞히는 network, noise 에서 거꾸로 돌아오는 reverse process, step 수와 DDIM, Diffusion Policy
papers:
  - title: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
    authors: Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli
    year: 2015
    url: https://arxiv.org/abs/1503.03585
  - title: "Denoising Diffusion Probabilistic Models"
    authors: Jonathan Ho, Ajay Jain, Pieter Abbeel
    year: 2020
    url: https://arxiv.org/abs/2006.11239
  - title: "Denoising Diffusion Implicit Models"
    authors: Jiaming Song, Chenlin Meng, Stefano Ermon
    year: 2020
    url: https://arxiv.org/abs/2010.02502
  - title: "Improved Denoising Diffusion Probabilistic Models"
    authors: Alex Nichol, Prafulla Dhariwal
    year: 2021
    url: https://arxiv.org/abs/2102.09672
  - title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    authors: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song
    year: 2023
    url: https://arxiv.org/abs/2303.04137
---

# DDPM

로봇이 차선 한가운데 놓인 상자를 피하는 시연을 모았더니, 절반은 핸들을 왼쪽으로 20° 꺾었고 나머지 절반은 오른쪽으로 20° 꺾었다. 이
시연으로 출력과 정답의 차이를 제곱해 줄이는 MSE 학습을 하면 장면을 보고 핸들 각도를 정하는 policy 는 두 정답의 평균인 0° 를 내서 상자를 정면으로 들이받는다. sample 을
뽑는 생성 모델인 VAE 로 바꿔도 sample 의 일부는 두 길 사이에 떨어진다. 정답이 둘로 갈라진 데이터에서 평균이 아닌 한쪽 길을 또렷하게
뽑으려면 무엇이 달라져야 할까?

DDPM(Denoising Diffusion Probabilistic Model)은 sample 을 한 번에 만들지 않고, 무작위로 뽑은 값인 noise 에서 출발해 noise 를 조금 걷어 내는 작은 step 을
수십에서 천 번 되풀이한다. 왼쪽과 오른쪽 가운데 어느 길로 갈지는 그 step 들을 지나는 동안 조금씩 정해진다. Chi 등의 Diffusion
Policy(2023)는 이 방법으로 로봇이 앞으로 할 action 여러 개(대부분의 task 에서 16 개)를 한꺼번에 만들었고, 네 benchmark 의 15 개 task 에서 이전 방법 가운데 가장
좋은 것보다 점수를 평균 46.9% 높였다(task 마다 높아진 비율의 평균).

## 학습 목표

- 한 번에 만드는 생성이 정답들의 평균을 내는 이유와, step 을 잘게 나눈 DDPM 이 정답이 갈라진 데이터에서 한 무리를 고르는 이유를 설명할 수 있다
- 데이터에 noise 를 섞어 가는 forward process 의 계수와, step 마다 섞는 양을 정한 noise schedule 로 어느 step 의 값이든 원래 데이터에서 계산 한 번으로 만들 수 있다
- network 가 섞인 noise 를 맞히도록 학습하는 이유와, 그 network 로 noise 에서 sample 을 만드는 reverse step 을 설명할 수 있다
- step 수가 sampling 시간을 정하는 이유와, step 을 건너뛰는 DDIM 과 Diffusion Policy 가 이 시간을 줄여 로봇에 쓰는 방식을 설명할 수 있다

## 한 번에 만드는 생성의 평균 문제

network 는 입력이 같으면 출력도 같다. 상자 앞이라는 같은 장면에 왼쪽 20° 와 오른쪽 20° 가 정답으로 섞여 있으면 network 는 장면만 보고 둘
중 하나를 고를 수 없고, 오차 제곱의 합을 줄이도록 학습하면 두 정답의 한가운데로 간다. 정답이 $-20$ 과 $+20$ 하나씩일 때 0 을 내면 오차
제곱의 합이 $400 + 400 = 800$ 이고, 10 을 내면 $900 + 100 = 1000$ 이고, $-20$ 을 내면 $0 + 1600 = 1600$ 이다. 출력이 0 에서 어느 쪽으로
벗어나도 합이 커지므로 0 이 loss 를 가장 작게 만든다.

[VAE](../vae/README.md) 의 decoder 도 계산 한 번으로 그림을 만드는 network 라서 같은 일을 겪는다. VAE 는 표준 정규분포에서 뽑은 숫자
몇 개(latent $z$)를 decoder 에 넣어 그림 하나를 내는데, 그 한 번의 계산 안에서 그림이 어느 무리에 속할지와 pixel 하나하나의 밝기를 함께 정해야
한다. 비슷한 $z$ 에 흰 pixel(1)과 검은 pixel(0)이 반반 짝지어져 학습되면 decoder 는 그 자리에 회색인 0.5 를 내고, VAE 의 sample 이 흐릿해
보이는 것도 이 때문이다.

network 를 키우거나 오래 학습해도 나아지지 않는다. MSE 로 보면 평균이 가장 좋은 답이라서, 학습이 잘될수록 평균에 더 가까이 간다.

## 잘게 나눠 되돌리는 생성

한 번의 계산이 문제라면 계산을 여러 번으로 나누면 되지만, 나눈 step 마다 network 가 풀 문제와 정답이 있어야 한다. 그런데 완성된 sample
로 가는 중간 상태의 정답은 데이터 어디에도 없다. 2015 년 Sohl-Dickstein 등은 이 중간 상태를 반대 방향에서 얻었다. 데이터에 noise 를
조금씩 더하면 구조가 천천히 지워지다가 결국 noise 만 남는데, 저자들은 이 망가뜨리는 과정을 잘게 나눠 정해 두고 그 step 하나를
거꾸로 되돌리는 법을 network 에게 배우게 했다. step 마다 망가뜨리기 전과 후의 값이 남으니 중간 상태의 문제와 정답이 따로 손대지 않아도
생긴다. 2020 년 Ho 등은 망가뜨리는 쪽의 noise 크기까지 미리 정한 상수로 두어 그쪽에서는 배울 것이 없게 만들었다.

되돌리는 step 하나하나가 다시 평균 문제에 걸리지 않으려면 step 을 잘게 나눠야 한다. 데이터가 $-2$ 와 $+2$ 두 값뿐인데 한 step 에 noise 를 크게 섞어 값이 0.2 가 되었다면,
섞기 전 값의 후보는 $-2$ 와 $+2$ 로 멀리 갈라지고 후보의 평균을 답하는 network 는 데이터에 없는 0 근처를 낸다. 한 step 에 섞는 noise 가
대개 $\pm 0.2$ 안쪽이라면, 섞은 뒤 값이 1.0 일 때 섞기 전 값은 대개 0.8 ~ 1.2 에 있었다. 후보가 지금 값 근처 한 군데에 모여 있으니 평균을
답해도 크게 틀리지 않는다. 저자들은 noise 가 작으면 후보가 이렇게 한 군데로 모인다는 것을 확률 이론의 결과(Feller, 1949)로 뒷받침했다.

새 데이터는 망가뜨리는 과정의 끝에서 출발해 한 step 씩 되돌아오며 만든다. 끝 상태는 noise 뿐이라 출발점을 매번 새로 뽑을 수 있고, 출발점이
다르면 같은 network 로도 다른 무리의 sample 에 닿는다. 잉크 한 방울이 물에 퍼지는 확산(diffusion)을 거꾸로 돌리는 것과 같아서 이 방법을
diffusion model 이라고 부른다.

![VAE 는 z 를 decoder 에 한 번 넣어 그림을 만들고, DDPM 은 noise 이미지 x_1000 에서 noise 를 조금씩 걷어 내 x₀ 를 만든다](figures/one-shot-vs-steps.svg)

그림의 $x$ 에 붙은 아래첨자는 step 번호다. $x_0$ 은 noise 가 없는 원래 데이터이고, 번호가 클수록 noise 가 많이 섞인 상태다. 데이터에서
noise 로 가는 방향은 **forward process** 라고 하고, noise 에서 데이터로 돌아오는 방향은 **reverse process** 라고 한다. DDPM 은 아래 세 부분으로 이루어진다.

| 부분 | 맡은 일 | 학습 |
| --- | --- | --- |
| [forward process](#forward-process) | 깨끗한 데이터에 noise 를 섞어, 섞인 값과 섞은 noise 를 짝지은 학습 문제를 만든다 | 계산으로 정해 두어 배우지 않는다 |
| [noise 를 맞히는 network](#noise-를-맞히는-network) | 섞인 값을 보고 그 안의 noise 를 맞힌다 | DDPM 에서 학습하는 부분은 이것 하나다 |
| [reverse process](#reverse-process-의-한-step) | noise 에서 출발해 network 가 맞힌 noise 를 조금씩 빼며 데이터로 돌아온다 | 학습한 network 를 되풀이해 쓴다 |

2020 년 Ho, Jain, Abbeel 은 그때까지 diffusion model 로 품질 높은 sample 을 만든 예가 없었다고 적은 뒤, 섞인 noise 를 맞히는 단순한 loss 로
선명한 그림을 만들어 보였다. 이 형태를 DDPM 이라고 부르는데, denoising 은 noise 를 걷어 낸다는 뜻이고 probabilistic 은 확률을 쓴다는 뜻이다.

잘게 나눈 대가로 sample 하나를 만들 때 network 를 step 수만큼 써야 한다. VAE 의 decoder 는 한 번이면 된다.

## forward process

망가뜨리는 규칙부터 정한다. 새 데이터를 만들 때는 원래 데이터를 알 수 없어서 출발점을 이 규칙의 끝 상태에서 뽑아야 하고, 그러려면 원래
데이터가 무엇이었든 끝 상태가 늘 같은 분포여야 한다. DDPM 은 그 분포로 평균 0, 표준편차 1 인 종 모양 분포인 표준 정규분포
$\mathcal N(0, 1)$ 을 고른다.

데이터에 섞는 noise 도 step 마다 $\mathcal N(0, 1)$ 에서 새로 뽑고, 그리스 문자 $\epsilon$("엡실론")으로 적는다.
[Probability & Gaussians](../probability/README.md) 의 68-95-99.7 규칙에 따라 뽑힌 $\epsilon$ 의 약 95% 는 $-2$ ~ 2 사이에 있다. $\epsilon$ 은
forward process 에서 데이터를 망가뜨리는 재료로 쓰이고, network 를 학습할 때는 network 가 맞힐 정답으로 다시 쓰인다. noise 는 우리가 직접
뽑아 섞으므로 무엇을 섞었는지 늘 안다.

noise 를 더하기만 하는 규칙부터 시험해 보자. $x_0 = 2$ 에 step 마다 $0.6 \times \epsilon$ 을 더하면 noise 가 0 을 중심으로 양쪽에 고르게
흩어지므로, 여러 번 실행한 값들의 가운데는 계속 2 에 머문다. 흩어진 정도인 분산만 step 마다 $0.6^2 = 0.36$ 씩 불어나 100 step 뒤에는 36 이 된다. $x_0 = -2$
에서 시작했다면 가운데가 $-2$ 에 머무니, 끝 상태만 보고도 원래 데이터를 알 수 있다.

가운데를 0 으로 보내려면 원래 값의 몫을 줄여야 해서, noise 를 더하기 전에 지금 값에 1 보다 작은 수를 곱한다. 0.8 을 곱하고
$0.6 \times \epsilon$ 을 더하는 규칙을 $x_0 = 2$ 에 다섯 번 적용하면 아래와 같다. $\epsilon$ 은 예로 고른 값이고, $x_{t-1}$ 은 step $t$ 를
시작할 때의 값이며 $x_t$ 는 step $t$ 를 마친 값이다.

| step $t$ | 지금 값 $x_{t-1}$ | $0.8 \times x_{t-1}$ | 뽑은 $\epsilon$ | $0.6 \times \epsilon$ | 새 값 $x_t$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1.6 | 0.5 | 0.3 | 1.9 |
| 2 | 1.9 | 1.52 | $-1$ | $-0.6$ | 0.92 |
| 3 | 0.92 | 0.74 | 0.2 | 0.12 | 0.86 |
| 4 | 0.86 | 0.69 | 1.5 | 0.9 | 1.59 |
| 5 | 1.59 | 1.27 | $-0.5$ | $-0.3$ | 0.97 |

$\epsilon$ 을 다르게 뽑은 아래 그림의 실행 2 는 같은 $x_0 = 2$ 에서 출발해도 전혀 다른 길로 간다.

![x₀ = 2 에서 ε 를 예로 골라 돌린 두 실행. 실행 1 은 1.90 → 0.92 → 0.86 → 1.59 → 0.97 로 가고, 실행 2 는 1.30 → 0.32 → 0.50 → −0.20 → 0.02 로 간다. 점선은 0.8 만 곱한 값이다](figures/forward-path.svg)

실행 하나는 noise 를 따라 오르내리지만, 같은 $x_0 = 2$ 에서 수많은 실행을 모으면 그 분포는 정해진 대로 움직인다. 분포의 가운데, 퍼진 정도,
모양이 각각 $\mathcal N(0, 1)$ 쪽으로 간다.

- 가운데(평균): 원래 값의 몫에는 step 마다 0.8 이 곱해져 $2 \times 0.8^t$ 로 줄고, noise 는 가운데를 옮기지 않으니 가운데가 0 으로 간다.
  $x_0 = -2$ 에서도 $-2 \times 0.8^t$ 라서 같은 0 에 닿는다
- 퍼진 정도(분산): 0.8 을 곱하면 분산이 $0.8^2 = 0.64$ 배로 줄고, noise 를 더하면 $0.6^2 = 0.36$ 이 는다. 분산이 1 보다 작을 때는 느는 양이
  줄어드는 양보다 커서 분산이 커지고, 1 이 되면 $0.64 \times 1 + 0.36 = 1$ 로 더 변하지 않는다. 0.8 과 0.6 은 $0.8^2 + 0.6^2 = 1$ 이 되게 고른 짝이라 분산이
  정확히 1 에서 멈춘다
- 모양: 종 모양 noise 를 여러 번 더한 값이라 종 모양이 된다

| step $t$ | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| 가운데 $2 \times 0.8^t$ | 1.6 | 1.28 | 1.02 | 0.66 | 0.21 | 0.023 |
| 분산 | 0.36 | 0.59 | 0.74 | 0.89 | 0.99 | 1.00 |

![x₀ = 2 에서 t = 1, 3, 10 의 분포. 더하기만 하면 가운데는 2 에 머물고 표준편차가 0.6, 1.04, 1.90 으로 커지며, 0.8 을 곱하고 더하면 가운데가 1.6, 1.02, 0.21 로 0 에 다가가고 표준편차는 0.6, 0.86, 0.99 로 1 에 다가간다](figures/add-vs-shrink.svg)

20 step 뒤에는 가운데가 0.023 이고 분산이 1.00 이라 $\mathcal N(0, 1)$ 과 구별하기 어렵다. 0.8 과 0.6 처럼 제곱해 더하면 1 이 되는 짝은 더하는
noise 의 분산 하나로 정해지므로, 규칙을 아래처럼 적는다.

$$
x_t = \sqrt{1 - \beta_t}\; x_{t-1} + \sqrt{\beta_t}\; \epsilon, \qquad \epsilon \sim \mathcal N(0, 1)
$$

$\beta_t$("베타")는 step $t$ 에서 더하는 noise 의 분산이다. $\epsilon \sim \mathcal N(0, 1)$ 은 "$\epsilon$ 을 $\mathcal N(0, 1)$ 에서 뽑는다"
로 읽는다. 위 예는 $\beta_t = 0.36$ 이라 $\sqrt{1 - 0.36} = 0.8$ 과 $\sqrt{0.36} = 0.6$ 이 나왔다. 어떤 $\beta_t$ 를 써도 두 계수의 제곱을 더하면
$(1 - \beta_t) + \beta_t = 1$ 이라서 분산은 언제나 1 에서 멈춘다.

> **오해** DDPM 의 $\beta_t$ 는 [VAE](../vae/README.md) 의 $\beta$ 와 같은 값이다.
>
> **사실** VAE 의 $\beta$ 는 loss 에서 KL 항에 곱하는 무게이고, DDPM 의 $\beta_t$ 는 step $t$ 에서 더하는 noise 의 분산이다. 같은 글자를 쓸
> 뿐 하는 일이 다르고, DDPM 쪽은 step 마다 값이 달라서 아래첨자 $t$ 가 붙는다.

<details><summary><b>심화: 분산을 1 로 지키는 계수</b></summary>

값에 $a$ 를 곱하면 분산은 $a^2$ 배가 되고, 서로 상관없이 따로 뽑은(독립인) 두 값을 더하면 분산끼리 더해진다. 평균 0, 분산 1 인 두 값
$-1$, $+1$ 에 0.8 을 곱하면 $-0.8$, $+0.8$ 이 되어 분산이 0.64 로 준다. 여기에 noise 쪽 값 $-0.6$, $+0.6$ 을 모든 짝으로 더하면 아래 그림
왼쪽처럼 $-1.4$, $-0.2$, $0.2$, $1.4$ 가 되고, 분산은 $(1.96 + 0.04 + 0.04 + 1.96) \div 4 = 1$ 로 $0.64 + 0.36$ 과 같다.

![왼쪽은 0.8 을 곱한 −0.8, +0.8 과 noise −0.6, +0.6 을 짝지어 더한 −1.4, −0.2, 0.2, 1.4 로 제곱의 평균이 1 이다. 오른쪽은 줄이지 않은 −1, +1 에 같은 noise 를 더한 −1.6, −0.4, 0.4, 1.6 으로 제곱의 평균이 1.36 이다](figures/variance-sum.svg)

줄이지 않고 더하면 분산이 $1 + 0.36 = 1.36$ 으로 커지고, 0.9 를 곱하고 $0.3\,\epsilon$ 을 더하면 $0.81 + 0.09 = 0.9$ 로 작아진다. 두 계수의
제곱을 더해 정확히 1 이 될 때만 분산이 1 로 유지된다.

</details>

이 규칙은 원래 값의 흔적을 지우려고 만든 것이라, 끝까지 밟고 나면 섞은 noise 를 모르는 한 원래 값으로 되돌아갈 수 없다. 되돌리는 데
필요한 그 noise 를 맞히는 일이 network 의 몫이다.

**확인 문제.** $\beta_t = 0.64$ 인 step 에서 $x_{t-1} = 2$ 이고 $\epsilon = 0.5$ 가 뽑혔다. $x_t$ 는?

<details><summary>답 보기</summary>

① 두 계수를 구한다.

$$
\sqrt{1 - 0.64} = \sqrt{0.36} = 0.6, \qquad \sqrt{0.64} = 0.8
$$

② 규칙에 넣는다.

$$
x_t = 0.6 \times 2 + 0.8 \times 0.5 = 1.2 + 0.4 = 1.6
$$

답: 1.6. $\beta_t$ 가 클수록 원래 값의 몫을 많이 덜고 noise 를 많이 넣는다.

</details>

## noise schedule

이제 network 에게 문제를 낸다. 새 데이터를 만들 때는 noise 뿐인 값에서 깨끗한 값까지 모든 step 을 지나므로 network 는 어느 step 의 값이든
풀 줄 알아야 하고, 그래서 학습 때마다 step $t$ 를 무작위로 골라 $x_t$ 를 문제로 낸다. 그런데 $t = 700$ 이 걸리면 곱하고 더하기를 700 번 해야
문제 하나가 나온다. forward process 에서 본 대로 $x_t$ 는 원래 값의 몫과 종 모양 noise 의 몫을 더한 값이므로, 두 몫의 크기만 미리 구해 두면
계산 한 번으로 만들 수 있다.

두 몫은 step 마다 섞은 양에서 나오므로 DDPM 은 step 수 $T$ 와 step 마다 섞을 $\beta_t$ 를 미리 정해 둔다. 이 표 $\beta_1, \ldots, \beta_T$ 를
**noise schedule** 이라고 한다. 앞쪽 step 에서는 $\beta_t$ 를 작게 두어 조금만 망가뜨리고 뒤로 갈수록 키운다. 이 강의에서는 손으로 셀 수
있게 예로 $T = 5$ 로 두고 $\beta_t$ 를 0.1 씩 늘린 schedule 을 쓴다.

표에는 기호가 둘 더 붙는다. forward process 의 0.8 은 $\sqrt{1 - 0.36}$ 이었는데, 이처럼 step $t$ 에서 원래 값의 몫에 곱하는 수
$\sqrt{1 - \beta_t}$ 의 제곱근 안쪽 $1 - \beta_t$ 를 $\alpha_t$("알파")로 적는다. step $t$ 까지 원래 값이 얼마나 남았는지는 이 수들을 step 1
부터 곱해야 알 수 있어서, $\alpha_1$ 부터 $\alpha_t$ 까지 곱한 값을 $\bar\alpha_t$("알파 바")로 적어 둔다.

| step $t$ | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| $\beta_t$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| $\alpha_t = 1 - \beta_t$ | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
| $\bar\alpha_t$ (바로 왼쪽 칸 × $\alpha_t$) | 0.9 | 0.72 | 0.504 | 0.3024 | 0.1512 |
| $\sqrt{\bar\alpha_t}$ (원래 값에 곱함) | 0.95 | 0.85 | 0.71 | 0.55 | 0.39 |
| $\sqrt{1 - \bar\alpha_t}$ (noise 에 곱함) | 0.32 | 0.53 | 0.70 | 0.84 | 0.92 |

![예로 쓰는 T = 5 schedule. β 는 0.1 ~ 0.5 로 커지고 ᾱ 는 0.9, 0.72, 0.504, 0.3024, 0.1512 로 작아진다](figures/schedule-t5.svg)

step 마다 $\sqrt{\alpha_t}$ 가 곱해지니 step $t$ 까지 원래 값에는 $\sqrt{\bar\alpha_t}$ 가 곱해지고, noise 몫의 분산은 $1 - \bar\alpha_t$ 가
된다. forward process 의 분산 표에서 0.36, 0.59, 0.74 가 $1 - 0.64$, $1 - 0.64^2$, $1 - 0.64^3$ 이었던 것과 같은 계산이다. 두 몫을 더하면
어느 step 의 값이든 원래 값에서 바로 만드는 식이 된다.

$$
x_t = \sqrt{\bar\alpha_t}\; x_0 + \sqrt{1 - \bar\alpha_t}\; \epsilon
$$

$x_0 = 2$, $t = 3$ 에서 $\epsilon = 0.5$ 가 뽑혔다면 표의 $t = 3$ 칸에서 $x_3 = 0.71 \times 2 + 0.70 \times 0.5 = 1.42 + 0.35 = 1.77$ 이다.
[VAE](../vae/README.md) 의 reparameterization trick $z = \mu + \sigma\epsilon$ 과 같은 꼴이라서, $x_t$ 는 평균 $\sqrt{\bar\alpha_t}\,x_0$,
표준편차 $\sqrt{1 - \bar\alpha_t}$ 인 정규분포를 따른다.

<details><summary><b>심화: 한 번에 건너뛰는 식의 유도</b></summary>

$\beta = 0.36$ 으로 고정하고 두 step 을 펼쳐 쓴다. $\epsilon_1$ 과 $\epsilon_2$ 는 step 1 과 step 2 에서 따로 뽑은 noise 다.

$$
\begin{aligned}
x_2 &= 0.8\, x_1 + 0.6\, \epsilon_2 = 0.8\,(0.8\, x_0 + 0.6\, \epsilon_1) + 0.6\, \epsilon_2 \\
    &= 0.64\, x_0 + 0.48\, \epsilon_1 + 0.6\, \epsilon_2
\end{aligned}
$$

$x_0 = 2$, $\epsilon_1 = 0.5$, $\epsilon_2 = -1$ 을 넣으면 $1.28 + 0.24 - 0.6 = 0.92$ 로, forward process 표에서 두 step 을 밟은 값과 같다.
noise 부분 $0.48\,\epsilon_1 + 0.6\,\epsilon_2$ 는 독립인 두 정규분포 값의 합이라 역시 정규분포이고, 분산은 $0.48^2 + 0.6^2 = 0.5904$ 다.
그래서 $\epsilon$ 하나에 $\sqrt{0.5904} = 0.768$ 을 곱한 것과 분포가 같아 $x_2 = 0.64\,x_0 + 0.768\,\epsilon$ 으로 쓸 수 있다.
$\bar\alpha_2 = 0.64 \times 0.64 = 0.4096$ 이므로 $\sqrt{\bar\alpha_2} = 0.64$ 와 $\sqrt{1 - \bar\alpha_2} = 0.768$ 이 바로 이 두 계수다.

![위는 x₀ = 2 에서 한 step 씩 밟아 x₁ = 1.9, x₂ = 0.92 를 얻는 계산이고, 아래는 같은 ε 로 0.64 × 2 = 1.28 에 noise 합 −0.36 을 더해 x₂ = 0.92 를 한 번에 얻는 계산](figures/jump.svg)

일반적으로는 $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ 로 적는다. $\prod$ 는 1 번째부터 $t$ 번째까지 모두 곱하라는 기호다.

</details>

schedule 을 짤 때는 $\beta_t$ 의 크기를 저울질해야 한다. $\beta_t$ 를 크게 잡으면 적은 step 으로 원래 값을 지울 수 있지만, 그만큼 step 하나가
크게 움직인다.

**확인 문제.** $T = 5$ schedule 에서 $x_0 = -2$, $t = 2$, $\epsilon = 1$ 이면 $x_2$ 는?

<details><summary>답 보기</summary>

① 표의 $t = 2$ 칸에서 $\sqrt{\bar\alpha_2} = 0.85$ 와 $\sqrt{1 - \bar\alpha_2} = 0.53$ 을 읽는다.

② 한 번에 건너뛰는 식에 넣는다.

$$
x_2 = 0.85 \times (-2) + 0.53 \times 1 = -1.70 + 0.53 = -1.17
$$

답: $-1.17$.

</details>

## T 와 β 의 크기

step 하나가 크게 움직이면 [잘게 나눠 되돌리는 생성](#잘게-나눠-되돌리는-생성) 에서 본 문제가 다시 나온다. 섞기 전 값의 후보가 한 군데에
모이는 것은 step 이 작을 때뿐이고, 크면 후보가 여러 무리로 갈라져 그 평균이 무리 사이에 떨어진다. 그래서 DDPM 은 step 을 1000 개로 나누고
$\beta_t$ 를 0.02 이하로 둔다.

$T = 5$ schedule 은 $\sqrt{\bar\alpha_5} = 0.39$ 라서 $x_5$ 에 원래 값이 아직 39% 남는다. 다섯 step 만에 원래 값을 거의 지우려면 step 마다
$\beta_t \approx 0.87$ 은 되어야 하는데, 데이터가 $-2$ 와 $+2$ 일 때 이렇게 큰 step 을 거친 값에서 직전 값을 짐작하면 후보가 $+2$ 근처와
$-2$ 근처 두 무리로 갈라진다. $\beta_t = 0.018$ 이면 한 step 에 표준편차 $\sqrt{0.018} \approx 0.13$ 만큼만 흔들려서 후보가 $x_t$ 바로 근처
한 군데에 몰리고, 되돌리는 쪽은 그 가운데와 폭만 맞히면 된다. 값이 몰린 자리는 아래 그림처럼 분포 그래프에서 산처럼 솟는데, 이렇게 솟은
부분을 **봉우리**라고 부른다.

![데이터가 −2 와 +2 일 때 x_t 를 보고 x_{t−1} 이 있을 범위. β = 0.87 인 큰 step 에서는 +2 근처(0.58)와 −2 근처(0.42) 두 무리로 갈라지고, β = 0.018 인 작은 step 에서는 표준편차 0.13 인 봉우리 하나다](figures/step-size.svg)

<details><summary><b>심화: 큰 step 에서 갈라지는 직전 값</b></summary>

$0.13^5 \approx 0.00004$ 이므로, $\bar\alpha_5$ 를 0.00004 정도로 줄이려면 step 마다 $\alpha_t = 0.13$ 이어야 하고 $\beta_t = 1 - 0.13 = 0.87$ 이 된다. 이런
step 한 번으로 $x_1 = \sqrt{0.13}\,x_0 + \sqrt{0.87}\,\epsilon = 0.36\,x_0 + 0.93\,\epsilon$ 을 만들었고 $x_1 = 0.2$ 가 나왔다고 하자. $x_0 = +2$
였다면 섞인 noise 는 $(0.2 - 0.72) \div 0.93 \approx -0.56$ 이고, $x_0 = -2$ 였다면 $(0.2 + 0.72) \div 0.93 \approx 0.99$ 다. 둘 다
$\mathcal N(0, 1)$ 에서 흔히 나오는 값이라, 두 경우의 확률 밀도를 합이 1 이 되게 나누면 $+2$ 쪽이 0.58 이고 $-2$ 쪽이 0.42 다. 이 범위를 종 모양
하나로 맞히면 가운데가 $0.58 \times 2 - 0.42 \times 2 \approx 0.33$ 에 놓인다.

</details>

Ho 등의 schedule 은 $T = 1000$ 에 $\beta_t$ 를 0.0001 에서 0.02 까지 고른 간격으로 늘린다(linear schedule). 가장 큰 마지막 step 도 noise 의
표준편차가 $\sqrt{0.02} \approx 0.14$ 라 step 하나가 작고, 1000 step 을 다 밟으면 $\bar\alpha_{1000} \approx 0.00004$ 라서 원래 값이 거의 남지
않는다. 이후 Nichol 과 Dhariwal 은 linear schedule 의 뒤쪽 4 분의 1 이 거의 noise 만 다룬다고 보고, $\bar\alpha_t$ 가 cos 곡선을 따라
천천히 줄어드는 cosine schedule 을 내놓았다.

<details><summary><b>심화: linear 와 cosine schedule 의 계수</b></summary>

두 schedule 의 계수를 계산하면 아래와 같다.

| $t$ | linear $\sqrt{\bar\alpha_t}$ | linear $\sqrt{1-\bar\alpha_t}$ | cosine $\sqrt{\bar\alpha_t}$ | cosine $\sqrt{1-\bar\alpha_t}$ |
| --- | --- | --- | --- | --- |
| 100 | 0.947 | 0.321 | 0.986 | 0.167 |
| 250 | 0.724 | 0.690 | 0.920 | 0.391 |
| 500 | 0.280 | 0.960 | 0.703 | 0.711 |
| 750 | 0.058 | 0.998 | 0.380 | 0.925 |
| 1000 | 0.006 | 1.000 | 0.000 | 1.000 |

![T = 1000 의 linear·cosine schedule 에서 x₀ 에 곱하는 √ᾱ_t(실선)와 noise 에 곱하는 √(1 − ᾱ_t)(점선). cosine 이 원래 값을 더 천천히 지운다](figures/schedule.svg)

</details>

로봇에서는 $T$ 를 더 작게 잡는다. Diffusion Policy 는 cosine schedule 을 $T = 100$ 으로 썼고, LeRobot 의 Diffusion Policy 설정도 기본값이
`num_train_timesteps: 100` 과 cosine schedule 인 `beta_schedule: squaredcos_cap_v2` 다. 이 schedule 의 $t = 50$ 에서
$\sqrt{\bar\alpha_{50}} \approx 0.70$ 이라, step 수만 줄었을 뿐 $T = 1000$ cosine 의 $t = 500$ 과 모양이 같다.

그림을 다룰 때는 pixel 밝기를 $-1$ ~ $+1$ 로 바꾸고 pixel 마다 $\epsilon$ 을 따로 뽑아 같은 식을 쓴다.

![8×8 그림 하나에 같은 noise 를 T = 1000 linear schedule 의 t = 0, 100, 250, 500, 1000 계수로 섞은 모습. t = 1000 에서는 noise 만 남는다](figures/image-noise.svg)

작게 나눈 만큼 sampling 은 길어진다. network 를 $T$ 번 써야 sample 하나가 나오므로 $T = 1000$ 이면 1000 번이고, Diffusion Policy 처럼
$T = 100$ 이어도 100 번이다.

**확인 문제.** $T = 1000$ 에서 $x_0 = 2$ 일 때, $x_{500}$ 에 남은 원래 값의 몫 $\sqrt{\bar\alpha_{500}} \times x_0$ 는 linear 와 cosine
schedule 에서 각각 얼마일까?

<details><summary>답 보기</summary>

① 표의 $t = 500$ 줄에서 linear $\sqrt{\bar\alpha_{500}} = 0.280$ 과 cosine $\sqrt{\bar\alpha_{500}} = 0.703$ 을 읽는다.

② 각각 2 를 곱한다.

$$
0.280 \times 2 = 0.56, \qquad 0.703 \times 2 \approx 1.41
$$

답: linear 는 0.56, cosine 은 약 1.41 이다. 같은 $t = 500$ 인데도 cosine 쪽에 원래 값이 두 배 넘게 남아 있다.

</details>

## noise 를 맞히는 network

문제를 만드는 법이 정해졌으니 network 가 맞힐 정답을 고를 차례다. 직전 값 $x_{t-1}$ 의 가운데를 맞히게 해도 되고, 원래 값 $x_0$ 나 섞인
noise $\epsilon$ 을 맞히게 해도 된다. Ho 등은 $x_0$ 를 바로 맞히게 하면 초기 실험에서 그림이 나빴다고 적었고, $\epsilon$ 을 맞히게 했을 때
가장 단순한 loss 로 가장 좋은 그림을 얻었다.

noise 를 맞히게 해도 원래 값은 언제든 계산할 수 있다. $x_t$ 는 원래 값과 noise 를 정해진 비율로 섞은 값이라서, $x_t$ 와 step 번호 $t$ 를
알면 둘 가운데 하나만 알아도 나머지가 나온다. [noise schedule](#noise-schedule) 의 $x_3 = 1.77$ 은 $0.71 \times x_0 + 0.70 \times \epsilon$
이므로, 섞인 noise 가 $\epsilon = 0.5$ 임을 알면 noise 몫 $0.70 \times 0.5 = 0.35$ 를 빼고 0.71 로 나눠 $x_0 = (1.77 - 0.35) \div 0.71 = 2$ 를
되찾는다. 기호로 쓰면 아래와 같다.

$$
x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\; \epsilon}{\sqrt{\bar\alpha_t}}
$$

![x₃ = 1.77 에서 섞인 noise 0.70 × 0.5 = 0.35 를 빼면 1.42 가 남고, 이것을 0.71 로 나누면 x₀ = 2.00 으로 돌아간다](figures/recover.svg)

noise 를 맞히는 network 는 $\epsilon_\theta(x_t, t)$ 로 적는다. $\theta$("세타")는 network 의 weight 전체이고, 괄호 안은 network 가 받는 입력이다.
출력은 맞힌 noise 라서 입력 $x_t$ 와 모양이 같다. 숫자 하나나 2 차원 점에는 MLP 를 쓰고, 그림에는 [CNN](../cnn/README.md) 의 convolution
층으로 그림을 줄였다가 다시 키워 pixel 마다 값을 내는 U-Net 을 쓴다. Diffusion Policy 가 action 에 쓰는 1D U-Net 은
[Diffusion Policy](../diffusion-policy/README.md) 에서 다룬다.

![숫자 하나와 2 차원 점은 MLP 에 넣고 32×32 이미지는 U-Net 에 넣는다. 어느 network 든 t 를 함께 받아 입력과 같은 모양의 noise 를 낸다](figures/network-io.svg)

noise 를 맞히게 하면 step 번호 $t$ 도 입력으로 줘야 한다. 같은 $x_t$ 라도 $t$ 에 따라 섞인 noise 가 달라서, $x_0 = 2$ 에서 온 $x_t = 1.42$ 는
$t = 3$ 이면 $1.42 = 0.71 \times 2 + 0.70 \times \epsilon$ 에서 $\epsilon = 0$ 이고, $t = 1$ 이면 $1.42 = 0.95 \times 2 + 0.32 \times \epsilon$
에서 $\epsilon = -1.5$ 다. $t$ 는 정해진 계산으로 숫자 여러 개짜리 vector(time embedding)로 바꿔 넣는다. LeRobot 의 Diffusion Policy 는 이
vector 를 작은 network 에 한 번 더 통과시켜 기본으로 숫자 128 개로 만든다.

## 학습과 loss

문제는 데이터마다 $t$ 와 noise 를 바꿔 끝없이 만들 수 있어서, 정답을 사람이 하나하나 달아야 한다면 학습이 불가능하다. DDPM 에서는 문제를
만들 때 섞을 noise 를 우리가 직접 뽑으므로 그 noise 를 그대로 정답으로 쓴다. Ho 등의 Algorithm 1 에 따른 학습 한 번은 아래와 같다.

1. 데이터에서 $x_0$ 하나를 꺼낸다
2. 1 부터 $T$ 사이에서 $t$ 를 무작위로 고른다
3. $\mathcal N(0, 1)$ 에서 $\epsilon$ 을 뽑아 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ 을 만든다
4. network 에 $x_t$ 와 $t$ 를 넣어 noise 를 맞히게 하고, 정답 $\epsilon$ 과의 차이를 제곱해 loss 를 구한다
5. gradient descent 로 loss 가 줄어드는 쪽으로 $\theta$ 를 조금 고친다

![학습 한 step. x₀ = 2, t = 3, ε = 0.5 로 x₃ = 1.77 을 만들고, network 가 예로 0.3 을 냈다고 하면 정답 0.5 와의 loss 는 0.04 다](figures/train-step.svg)

그림처럼 noise 가 숫자 하나이고 network 가 0.3 을 냈다면 loss 는 $(0.5 - 0.3)^2 = 0.04$ 다. noise 가 숫자 여러 개이면 자리마다 차이를 제곱해
더하고, 예 여러 개를 한 번에 학습할 때는 예마다 구한 loss 를 평균한다. 순서 3 이 한 번에 건너뛰는 식이라 $t = 700$ 이 뽑혀도 step 을 차례로 밟지
않는다. 학습을 오래 하는 동안 $t$ 가 고르게 뽑히므로 network 는 모든 step 을 연습하게 된다.

<details><summary><b>심화: 여러 예의 loss</b></summary>

데이터가 $+2$ 와 $-2$ 두 값이고 $T = 5$ schedule 을 쓸 때 예 세 개를 계산하면 아래와 같다. network 의 예측 칸에는 학습 도중에 나올 법한 값을
예로 골라 넣었다.

| 예 | $x_0$ | $t$ | $\epsilon$ | $x_t$ | network 의 예측 | loss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 0.5 | $0.71 \times 2 + 0.70 \times 0.5 = 1.77$ | 0.3 | $(0.5 - 0.3)^2 = 0.04$ |
| 2 | $-2$ | 1 | $-1$ | $0.95 \times (-2) + 0.32 \times (-1) = -2.22$ | $-0.6$ | $(-1 + 0.6)^2 = 0.16$ |
| 3 | 2 | 5 | 1.2 | $0.39 \times 2 + 0.92 \times 1.2 \approx 1.88$ | 1.0 | $(1.2 - 1.0)^2 = 0.04$ |

평균은 $(0.04 + 0.16 + 0.04) \div 3 = 0.08$ 이고, 기호로는 $L = \frac{1}{N}\sum_{i=1}^{N} \lVert \epsilon_i - \epsilon_\theta(x_{t_i}, t_i) \rVert^2$
로 적는다. $N$ 은 예의 개수이고, $\sum$ 은 1 번째부터 $N$ 번째까지 더하라는 기호이며, $\lVert \cdot \rVert^2$ 은 자리마다 차이를 제곱해 더한 값이다.
위 표에서는 $N = 3$ 이고 $L = 0.08$ 이다.

</details>

noise 를 정답으로 쓰면 출력의 크기도 고르다. $t$ 가 1 이든 1000 이든 정답은 $\mathcal N(0, 1)$ 에서 뽑은 숫자라서, network 하나가 모든 step
에서 비슷한 크기의 값을 내면 된다. LeRobot 의 Diffusion Policy 도 기본 설정 `prediction_type: epsilon` 으로 noise 를 맞히고, 이를 `sample`
로 바꾸면 깨끗한 action 을 직접 맞힌다.

<details><summary><b>심화: ELBO 와 단순한 loss</b></summary>

DDPM 의 loss 는 VAE 처럼 데이터가 나올 확률에 log 를 씌운 값의 하한(ELBO)에서 유도할 수 있다. 유도하면 step 마다 다른 무게 $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}$ 를 곱한
noise 오차의 합이 나오는데, Ho 등은 이 무게를 떼고 모든 step 을 같은 무게로 더한 $L_\text{simple}$ 을 썼다. 무게 속 $\sigma_t$ 는 sample 을 만들 때 step 마다 새로 더하는 noise 의 표준편차다. 무게를 떼면
noise 가 작아 쉬운 작은 $t$ 의 비중이 줄어서, 어려운 큰 $t$ 의 오차를 더 많이 줄이게 된다. 저자들이 32×32 컬러 사진 dataset 인 CIFAR-10 으로 한 실험에서 이 단순한 loss 가 FID 3.17 로 가장 좋았다. FID 는 만든
그림과 실제 그림의 통계가 얼마나 다른지 잰 점수라서 낮을수록 실제 그림에 가깝다.

</details>

이렇게 학습해도 loss 는 0 까지 내려가지 않는다.

> **오해** 학습이 잘 되면 loss 가 0 까지 내려간다.
>
> **사실** 같은 입력에 정답이 여럿이면 loss 는 0 이 될 수 없다. 데이터가 $+2$ 와 $-2$ 이고 $t = 5$ 에서 $x_5 = 0.2$ 를 받았다면, 섞인 noise
> 는 원래 값이 $+2$ 일 때 $(0.2 - 0.78) \div 0.92 \approx -0.63$ 이고 $-2$ 일 때 $(0.2 + 0.78) \div 0.92 \approx 1.07$ 이다. network 가 무엇을
> 내든 둘 가운데 하나와는 어긋나고, 이때 network 가 내는 값은 [1 차원 두 봉우리](#1-차원-두-봉우리) 에서 계산한다.

## reverse process 의 한 step

학습을 마치면 새 데이터를 만든다. 이번에는 원래 데이터도 섞인 noise 도 모르고, 아는 것은 forward process 의 끝 상태가 $\mathcal N(0, 1)$
이라는 사실뿐이다. forward process 를 그렇게 짜 둔 덕분에 $\mathcal N(0, 1)$ 에서 뽑은 $x_T$ 를 끝 상태로 삼아 출발할 수 있고, 거기서부터
network 가 맞힌 noise 로 forward process 를 한 step 씩 거꾸로 밟는다.

forward 한 step 이 직전 값에 $\sqrt{\alpha_t}$ 를 곱하고 noise 를 더한 것이었으니, 거꾸로는 이번 step 에 더해진 noise 를 빼고 $\sqrt{\alpha_t}$
로 나눈다. network 가 맞히는 noise 는 $x_0$ 부터 쌓인 noise 전체라서 그 가운데 이번 step 의 몫만 뺀다. Ho 등의 Algorithm 2 가 이 계산이다.

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\;\epsilon_\theta(x_t, t)\Big) + \sigma_t\, z,
\qquad z \sim \mathcal N(0, 1)
$$

"network 가 맞힌 noise 에 이번 step 의 계수를 곱해 $x_t$ 에서 빼고 $\sqrt{\alpha_t}$ 로 나눈 뒤, 새로 뽑은 $z$ 에 $\sigma_t$ 를 곱해 더한다"
로 읽는다. 여기서 $z$ 는 $\mathcal N(0, 1)$ 에서 새로 뽑은 숫자로, VAE 의 latent $z$ 와는 관계가 없다. $T = 5$ schedule 에서 계수는 아래와 같다.

| step $t$ | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| $\beta_t / \sqrt{1 - \bar\alpha_t}$ (빼는 noise 의 계수) | 0.32 | 0.38 | 0.43 | 0.48 | 0.54 |
| $1 / \sqrt{\alpha_t}$ (나눠 되돌리는 수) | 1.05 | 1.12 | 1.20 | 1.29 | 1.41 |
| $\sigma_t = \sqrt{\beta_t}$ (새 noise 의 크기) | 0.32 | 0.45 | 0.55 | 0.63 | 0.71 |

![reverse step 한 번. x₅ = 0.60 → 0.54 × 0.23 = 0.12 빼기 → 0.48 → 1.41 곱하기 → 0.68 → 새 noise 0.21 더하기 → x₄ = 0.89](figures/reverse-step.svg)

새 noise $\sigma_t z$ 를 더하는 것은 되돌린 값이 하나로 정해지지 않기 때문이다. 괄호 안의 계산은 $x_{t-1}$ 이 있을 법한 범위의
가운데를 내고, 같은 $x_t$ 에 이를 수 있는 $x_{t-1}$ 은 그 가운데 근처에 종 모양으로 퍼져 있다. 가운데에 새 noise 를 더해야 그 범위에서 하나를
뽑을 수 있고, 가운데만 따라가면 sample 이 실제 데이터보다 좁게 몰린다. 데이터가 $\pm 2$ 에 표준편차 0.5 로 퍼져 있을 때, 이 강의에서
noise 를 정확히 맞히는 network 로 $T = 1000$ sampling 을 2 만 번 해 보면 새 noise 를 더하지 않은 경우 $+$ 쪽 sample 의 표준편차가 0.5 에서 0.14 로
줄어 봉우리 안쪽에 몰린다.

![데이터가 ±2 에 표준편차 0.5 로 퍼진 두 봉우리일 때 이 강의에서 만든 sample 20000 개의 histogram. 새 noise 를 더하면 + 쪽 sample 이 평균 2.00, 표준편차 0.50 으로 데이터와 겹치고, 더하지 않으면 평균 1.45, 표준편차 0.14 로 좁게 몰린다](figures/noise-z.svg)

마지막 step($t = 1$)이 내는 값은 최종 sample $x_0$ 라서, 깨끗하게 만든 값을 다시 흐트러뜨리지 않도록 $z$ 를 더하지 않는다. 데이터가 $+2$
와 $-2$ 두 값일 때 $x_5 = 0.60$ 에서 출발한 sampling 은 아래 그림의 실행 A 처럼 2.00 에 닿는다. step 4 ~ 2 에서 network 가 낸 음수
noise 는 $x_t$ 에 음수 noise 가 섞여 값이 원래보다 작아졌다는 뜻이라, 그 몫을 빼면 값이 $+2$ 쪽으로 올라간다.

![z 를 예로 골라 돌린 T = 5 sampling 두 번. 실행 A 는 0.60 에서 출발해 2.00 에 닿고, 실행 B 는 −0.40 에서 출발해 −0.73, −0.88, −1.33, −1.83 을 거쳐 −2.00 에 닿는다](figures/sampling-runs.svg)

<details><summary><b>심화: T = 5 sampling 계산표</b></summary>

실행 A 를 한 줄씩 계산한 표다. $\epsilon_\theta$ 칸에는 noise 를 정확히 맞히는 network 가 낼 값을 [1 차원 두 봉우리](#1-차원-두-봉우리) 의
방법으로 구해 넣었고, $z$ 칸에는 예로 고른 값을 넣었다. 곱한 결과는 소수 셋째 자리에서 반올림했다.

| $t$ | $x_t$ | $\epsilon_\theta$ | 계수 × $\epsilon_\theta$ | 빼고 $\times \frac{1}{\sqrt{\alpha_t}}$ | $z$ | $\sigma_t z$ | $x_{t-1}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.60 | 0.23 | $0.54 \times 0.23 = 0.12$ | $1.41 \times 0.48 = 0.68$ | 0.3 | 0.21 | 0.89 |
| 4 | 0.89 | $-0.10$ | $-0.05$ | $1.29 \times 0.94 = 1.21$ | $-0.2$ | $-0.13$ | 1.08 |
| 3 | 1.08 | $-0.47$ | $-0.20$ | $1.20 \times 1.28 = 1.54$ | 0.2 | 0.11 | 1.65 |
| 2 | 1.65 | $-0.09$ | $-0.03$ | $1.12 \times 1.68 = 1.88$ | 0.2 | 0.09 | 1.97 |
| 1 | 1.97 | 0.23 | 0.07 | $1.05 \times 1.90 = 2.00$ | 없음 | 0 | 2.00 |

실행 B 는 $x_5 = -0.40$ 에서 $\epsilon_\theta = -0.14$, 0.20, 0.74, 0.69, 0.21 과 $z = -0.4$, 0.3, 0.2, $-0.1$ 로 같은 계산을 해 $-2.00$ 에 닿는다.

</details>

<details><summary><b>심화: 빼는 noise 의 계수</b></summary>

network 가 맞힌 noise 전체는 $\sqrt{1 - \bar\alpha_t}\,\epsilon$ 이고, 그 분산 $1 - \bar\alpha_t$ 가운데 이번 step 이 보탠 분산은 $\beta_t$ 다.
그래서 쌓인 noise 의 $\frac{\beta_t}{1 - \bar\alpha_t}$ 만큼을 이번 step 의 몫으로 보고 빼는데, 여기에 noise 전체의 계수 $\sqrt{1 - \bar\alpha_t}$ 를
곱해 약분하면 $\frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}$ 가 된다.

| step $t$ | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| $\beta_t \div (1 - \bar\alpha_t)$ | $0.1 \div 0.1 = 1.00$ | $0.2 \div 0.28 = 0.71$ | $0.3 \div 0.496 = 0.60$ | 0.57 | 0.59 |

step 1 에서는 noise 전체가 step 1 에서 더해진 것이라 몫이 1 이고, 이때 식은 forward 한 step $x_1 = \sqrt{0.9}\,x_0 + \sqrt{0.1}\,\epsilon$ 을
그대로 거꾸로 푼 것과 같다. 이 계수는 몫을 나누는 직관으로 얻었지만, $x_t$ 와 $x_0$ 를 함께 알 때 $x_{t-1}$ 이 따르는 정규분포의 평균을 정확히
구해도 같은 식이 나온다(Ho 등의 식 (7) 과 (11)).

</details>

**확인 문제.** $T = 5$ schedule 의 step 2 에서 $x_2 = 1.0$ 이고 network 가 $\epsilon_\theta = 0.5$ 를 냈다. 새 noise 를 더하기 전의 값은?

<details><summary>답 보기</summary>

① 계수 표의 step 2 칸에서 빼는 noise 의 계수 0.38 과 나눠 되돌리는 수 1.12 를 읽는다.

② 이번 step 의 몫을 뺀다.

$$
1.0 - 0.38 \times 0.5 = 1.0 - 0.19 = 0.81
$$

③ 1.12 를 곱한다.

$$
0.81 \times 1.12 \approx 0.91
$$

답: 약 0.91. 여기에 $\sigma_2 z = 0.45 \times z$ 를 더하면 $x_1$ 이 된다.

</details>

## 1 차원 두 봉우리

network 는 MSE 로 학습했으니 같은 입력에 정답이 여럿이면 평균을 낸다. 그렇다면 서론의 상자 장면에서 DDPM 도 가운데 길을 낼 것 같지만, 실제로는
한쪽 길을 낸다. 평균이 봉우리 사이로 끌려가는 일은 sampling 초반의 noise 가 큰 step 에서만 일어나고, 거기서 나온 값은 최종 sample 이 아니라
다음 step 의 출발점으로만 쓰이기 때문이다.

실행 A 의 첫 step 에서 network 가 낸 0.23 은 데이터가 $+2$ 와 $-2$ 반반일 때 network 가 무엇을 배우는지 따져 보면 계산할 수 있다. 먼저 $x_5 = 0.2$
를 받은 경우를 보자. 이 값이 $+2$ 에서 왔다면 $x_5$ 는 평균 $0.39 \times 2 = 0.78$, 표준편차 0.92 인 정규분포를 따르고, $-2$ 에서 왔다면
평균이 $-0.78$ 이다. 0.2 에서 두 곡선의 높이는 0.36 과 0.25 라서, 0.2 가 $+2$ 에서 왔을 가능성은 $0.36 \div (0.36 + 0.25) \approx 0.59$ 이고
$-2$ 에서 왔을 가능성은 0.41 이다.

![t = 5 에서는 +2 와 −2 에서 온 x₅ 의 분포가 겹쳐 0.2 에서의 높이가 0.36 과 0.25 이고, t = 2 에서는 떨어져 있어 1.0 에서의 높이가 0.32 와 0.00 이다](figures/posterior-bells.svg)

network 는 가능한 정답들을 이 가능성으로 섞은 평균을 내므로, 원래 값을 $0.59 \times 2 + 0.41 \times (-2) = 0.36$ 으로 짐작하고 그에 맞는
noise 를 낸다. $t = 5$ 에서는 두 곡선이 많이 겹쳐 짐작이 0 쪽으로 끌려가지만, noise 가 작은 $t = 2$ 에서는 두 곡선이 떨어져 있어 같은
0.2 라도 $+2$ 쪽 가능성이 0.92 로 커진다. 실행 A 의 $x_5 = 0.60$ 에서는 $+2$ 쪽 가능성이 0.75 라 원래 값을
$0.75 \times 2 - 0.25 \times 2 = 1.0$ 으로 짐작하고, 섞인 noise 를 $(0.60 - 0.39 \times 1.0) \div 0.92 \approx 0.23$ 으로 낸다.

<details><summary><b>심화: 가능성과 추정의 표</b></summary>

$+2$ 에서 온 $x_t$ 는 평균 $2\sqrt{\bar\alpha_t}$, 표준편차 $\sqrt{1 - \bar\alpha_t}$ 인 정규분포를 따르고 $-2$ 에서 온 $x_t$ 는 평균
$-2\sqrt{\bar\alpha_t}$ 인 정규분포를 따른다. 두 곡선의 높이를 합이 1 이 되게 나눈 것이 가능성이고, 원래 값의 추정은
$2 \times (+2 \text{ 쪽 가능성}) - 2 \times (-2 \text{ 쪽 가능성})$ 이다.

| $t$ | $x_t$ | $+2$ 에서 왔을 가능성 | $-2$ 에서 왔을 가능성 | 원래 값의 추정 |
| --- | --- | --- | --- | --- |
| 5 | 0.2 | 0.59 | 0.41 | 0.36 |
| 5 | 1.0 | 0.86 | 0.14 | 1.44 |
| 2 | 0.2 | 0.92 | 0.08 | 1.68 |
| 2 | 1.0 | 1.00 | 0.00 | 2.00 |

$t = 2$ 에서 $x_2 = 1.0$ 은 $+2$ 쪽 분포(평균 $0.85 \times 2 = 1.70$, 표준편차 0.53)의 평균에서 표준편차 1.3 개쯤 떨어진 흔한 값이지만,
$-2$ 쪽 분포에서는 평균 $-1.70$ 에서 2.7 떨어져 표준편차 5 개가 넘으므로 거의 나오지 않는다.

</details>

$T = 1000$ 으로 늘려 경로 여러 개를 그려 보면 sample 이 봉우리 하나를 고르는 과정이 보인다. $t$ 가 클 때 network 는 원래 값을 0 근처로
짐작해 경로를 어느 쪽으로도 세게 당기지 않고, 경로는 새 noise 에 따라 흔들린다. 그러다 우연히 조금 위로 간 경로는 짐작이 $+2$ 쪽으로
기울어 다음 step 에서 그쪽으로 더 끌려가고, 갈수록 확실하게 그 봉우리로 간다.

![데이터가 −2 와 +2 일 때 이 강의에서 계산한 T = 1000 sampling 경로 30 개. 0 근처에서 섞여 있다가 갈라져 두 점에 닿고, 최종값 4000 개 중 51% 가 +2, 49% 가 −2 다](figures/paths-1d.svg)

이 강의에서 noise 를 정확히 맞히는 network 로 sample 4000 개를 만들자 51% 는 $+2$ 에서 끝나고 49% 는 $-2$ 에서 끝났으며, $-1.5$ ~ 1.5 사이에서 끝난
sample 은 하나도 없었다. 같은 데이터에서 MSE 로 정답 하나를 배우는 모델은 언제나 0 을 낸다.

> **오해** network 가 평균을 내니 DDPM 도 봉우리 사이의 값을 낸다.
>
> **사실** $t$ 가 작아지면 $x_t$ 만 보고도 어느 봉우리에서 왔는지 알 수 있어서 network 가 내는 평균이 한 봉우리의 값과 같아진다.

이 성질은 봉우리를 고르는 일이 여러 step 에 나뉘어 일어나기 때문에 생기므로, step 을 너무 줄이면 사라진다. 앞쪽 step 의 짐작은 여전히 두
봉우리의 평균이라, 뒤쪽 step 을 건너뛰고 그 짐작을 최종 sample 로 쓰면 평균이 그대로 남는다.

## VAE 와 DDPM 의 비교

같은 데이터로 VAE 를 학습해 보면 이 성질이 얼마나 쓸모 있는지 보인다. VAE 의 decoder 는 weight 를 곱하고 더하는 계산이라 $z$ 가 조금
바뀌면 출력도 조금만 바뀌어서, 한 값에서 다른 값으로 건너뛸 수 없다. latent 가 숫자 하나이고 데이터가 $-2$ 와 $+2$ 일 때 decoder 를
$z \le -0.25$ 에서는 $-2$ 를 내고 $z \ge 0.25$ 에서는 $+2$ 를 내도록 잘 학습시켰어도 그 사이에서는 출력이 $-2$ 에서 $+2$ 까지 이어져야 한다.
$\mathcal N(0, 1)$ 에서 뽑은 $z$ 가 $-0.25$ ~ 0.25 에 들어갈 확률이 약 20% 라서, sample 의 약 20% 가 두 봉우리 사이에 떨어진다.

![예로 든 1 차원 decoder. z 가 −0.25 ~ 0.25 사이일 때 −2 와 +2 사이를 낸다. N(0, 1) 에서 뽑은 z 의 약 20% 가 이 띠에 들어간다](figures/vae-decoder-1d.svg)

<details><summary><b>심화: 2 차원 데이터의 비교</b></summary>

2 차원 데이터로도 비교해 보았다. 반지름 2 인 원 위에 45° 간격으로 중심 여덟 개를 놓고 중심마다 표준편차 0.1 로 데이터를 모았다. 여기에 latent 가
2 개인 VAE 와 $T = 1000$ 인 DDPM 을 hidden 층 3 개짜리 MLP 로 학습시켜 sample 을 2000 개씩 뽑았다. 중심에서 0.3 넘게 벗어난 sample 은 실제
데이터에서 약 1% 인데, VAE 에서는 약 22% 로 늘었고 DDPM 에서는 약 2% 에 그쳤다. DDPM 의 sample 은 봉우리마다 11 ~ 15% 씩 들어가, 여덟 봉우리에 고르게 나뉠 때의 12.5% 에 가까웠다.

![봉우리 8 개인 2 차원 데이터와 이 강의에서 학습한 VAE, DDPM 의 sample. 봉우리에서 0.3 넘게 벗어난 비율은 데이터, VAE, DDPM 이 각각 약 1%, 22%, 2% 다](figures/toy-compare.svg)

</details>

계산량은 반대라서, sample 하나를 만들 때 VAE 는 decoder 를 한 번 쓰지만 DDPM 은 network 를 $T$ 번 쓴다.

**확인 문제.** 1 차원 VAE 의 decoder 가 $z = -1$ 에서 $-2$ 를 내고 $z = 1$ 에서 $+2$ 를 내며, 그 사이를 직선으로 잇는다. $\mathcal N(0, 1)$ 에서
뽑은 $z$ 로 sample 을 만들면 대략 몇 % 가 두 봉우리 사이에 놓일까?

<details><summary>답 보기</summary>

① 두 봉우리 사이의 출력은 $z$ 가 $-1$ 과 1 사이일 때 나온다.

② 68-95-99.7 규칙에 따라 $\mathcal N(0, 1)$ 의 값이 $-1$ ~ 1 에 들어갈 확률은 약 68% 다.

답: 약 68%. decoder 가 봉우리 사이를 완만하게 이을수록 봉우리 사이 sample 이 늘어난다.

</details>

## step 수와 sampling 시간

학습은 예 하나에 network 를 한 번 쓰지만, sampling 은 정답 noise 를 모르는 채 $t = T$ 부터 1 까지 모든 step 을 차례로 밟아야 해서 sample
하나에 network 를 $T$ 번 쓴다. sampling 을 빠르게 하는 방법인 DDIM 의 논문에 따르면 Nvidia 2080 Ti GPU 로 DDPM 의 $32 \times 32$ 그림 5 만 장을 만드는 데 약 20 시간이
걸리지만, 한 번에 그림을 만드는 GAN 은 1 분도 걸리지 않는다.

로봇에서는 sampling 이 끝나야 action 이 나오므로 이 시간이 그대로 반응 지연이 된다. Diffusion Policy 의 실물 실험에서 policy 는 0.1
초마다 손끝 목표를 냈는데, network 한 번에 0.01 초가 걸린다고 하면 $T = 1000$ 은 sample 하나에 10 초가 걸리고 $T = 100$ 도 1 초가 걸린다.

같은 reverse step 식으로 여러 step 을 한 번에 건너뛰면 빨라지기는 하지만 품질이 떨어진다. 이 식은 직전 값이 봉우리 하나짜리 종 모양으로
퍼져 있다고 보고 만들었는데, 10 step 씩 건너뛰면 한 번에 바뀌는 양이 커져 후보가 여러 무리로 갈라진다. 그러면 [T 와 β 의 크기](#t-와-β-의-크기)
에서 본 문제가 돌아온다.

**확인 문제.** network 한 번에 0.02 초가 걸린다. $T = 100$ 으로 모든 step 을 밟을 때와 10 step 으로 줄였을 때 sample 하나에 걸리는 시간은?

<details><summary>답 보기</summary>

network 를 쓰는 횟수에 한 번의 시간을 곱한다.

$$
100 \times 0.02 = 2 \text{ 초}, \qquad 10 \times 0.02 = 0.2 \text{ 초}
$$

답: 2 초와 0.2 초.

</details>

## DDIM

DDPM 이 나온 2020 년에 Song, Meng, Ermon 은 학습은 그대로 두고 sampling 만 빠르게 하는 DDIM(Denoising Diffusion Implicit Models)을
내놓았다. DDIM 은 network 가 맞힌 noise 로 원래 값을 먼저 어림한 뒤, 그 어림을 [noise schedule](#noise-schedule) 의 한 번에 건너뛰는 식에
넣어 원하는 step 으로 곧장 간다. 한 번 건너뛸 때 계산은 둘이다.

1. 맞힌 noise 로 원래 값의 추정 $\hat x_0$("엑스 제로 햇")를 구한다: $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon_\theta) \div \sqrt{\bar\alpha_t}$
2. 그 $\hat x_0$ 와 같은 $\epsilon_\theta$ 를 한 번에 건너뛰는 식에 넣어, 원하는 더 작은 step $s$ 의 값을 만든다:
   $x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1 - \bar\alpha_s}\,\epsilon_\theta$

새 noise $z$ 를 더하지 않으므로 출발점이 같으면 sample 도 늘 같다. $T = 5$ 예에서 $x_5 = 0.60$ 으로 시작해 $t = 2$ 를 거쳐 $t = 0$ 으로 두
번만 건너뛰면 아래와 같다. $\epsilon_\theta$ 는 실행 A 와 같은 network 의 값이다.

| 건너뛰기 | $x_t$ | $\epsilon_\theta$ | $\hat x_0$ | 다음 값 |
| --- | --- | --- | --- | --- |
| $5 \to 2$ | 0.60 | 0.23 | $(0.60 - 0.92 \times 0.23) \div 0.39 \approx 1.00$ | $0.85 \times 1.00 + 0.53 \times 0.23 \approx 0.97$ |
| $2 \to 0$ | 0.97 | $-1.37$ | $(0.97 + 0.53 \times 1.37) \div 0.85 \approx 2.00$ | $\bar\alpha_0 = 1$ 이라 $\hat x_0$ 그대로 2.00 |

network 를 두 번만 쓰고 $+2$ 에 닿았다. 반면 $t = 5$ 에서 $t = 0$ 으로 한 번에 건너뛰면 첫 줄의 $\hat x_0 = 1.00$ 이 그대로 sample 이 되는데,
1.00 은 두 봉우리를 가능성으로 섞은 평균이라 데이터에 없는 값이다. [1 차원 두 봉우리](#1-차원-두-봉우리) 에서 본 대로, step 을 지나치게
줄이면 평균이 남는다.

![x₅ = 0.60 에서 시작한 세 sampling. DDPM 5 step 과 DDIM 두 번의 건너뛰기(0.97 경유)는 2.00 에서 끝나고, 한 번에 건너뛰면 두 봉우리의 평균 1.00 에서 끝난다](figures/ddim-jump.svg)

32×32 컬러 사진 dataset 인 CIFAR-10 으로 한 DDIM 논문의 실험 결과는 아래와 같다. 숫자는 만든 그림과 실제 그림의 통계가 얼마나 다른지 잰 FID 라서 낮을수록 실제 그림에 가깝다. 둘째 행은 같은 network 로 DDPM 의 reverse
step 식을 써서 건너뛴 결과다. 이 행의 식은 새 noise 의 분산을 이 강의의 $\beta_t$ 보다 조금 작게 둔다(한 step 씩 밟을 때 $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$). Ho 등은 한
step 씩 밟을 때는 두 값의 결과가 비슷했다고 적었다.

| sampling step 수 | 10 | 20 | 50 | 100 | 1000 |
| --- | --- | --- | --- | --- | --- |
| DDIM | 13.36 | 6.84 | 4.67 | 4.16 | 4.04 |
| DDPM 식으로 건너뜀 | 41.07 | 18.36 | 8.01 | 5.78 | 4.73 |

DDIM 은 20 ~ 100 step 으로 1000 step 과 비슷한 품질을 내고, 저자들은 이를 원래 DDPM 보다 10 ~ 50 배 빠르다고 적었다. 그래도
100 step 에서 10 step 으로 줄이면 network 를 쓰는 횟수는 10 분의 1 이 되지만 FID 가 4.16 에서 13.36 으로 나빠진다.

Diffusion Policy 논문 본문에 따르면 저자들은 실물 로봇에서 100 step schedule 로 학습한 network 로 DDIM 10 step 만 밟아, Nvidia RTX 3080
에서 action 을 0.1 초 만에 만들었다. 같은 논문의 부록(A.4 절과 표 7)에는 실물 실험의 sampling step 수가 16 으로 적혀 있다.

<details><summary><b>심화: DDIM 의 일반 식</b></summary>

DDIM 논문은 건너뛸 때 새 noise 를 얼마나 더할지를 $\eta$("에타")로 정한다.

$$
x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1 - \bar\alpha_s - \sigma^2}\;\epsilon_\theta + \sigma z,
\qquad \sigma = \eta \sqrt{\frac{1 - \bar\alpha_s}{1 - \bar\alpha_t}} \sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_s}}
$$

$\eta = 0$ 이면 $\sigma = 0$ 이라 본문의 식이 된다. $\eta = 1$ 이면 한 step 씩 밟을 때 새 noise 의 분산을
$\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$ 로 둔 DDPM 의 reverse step 과 같은 분포가 되고, 위 표의 "DDPM 식으로 건너뜀" 행이 이
경우다. 이 강의처럼 새 noise 의 분산을 $\beta_t$ 로 두고 건너뛰면 결과가 더 나빠서, DDIM 논문의 같은 표에서 10 step 의 FID 가 367.43 이다.
$\eta = 0$ 의 sampling 은 noise 에서 데이터로 가는 정해진 매끄러운 경로 하나를 따라가므로 큰 step 에도 오차가 덜 쌓인다. 이 경로를
처음부터 곧게 학습하는 방법은 [Flow Matching](../flow-matching/README.md) 에서 다룬다.

</details>

**확인 문제.** $T = 5$ schedule 에서 $x_5 = 1.0$ 이고 network 가 $\epsilon_\theta = 0.5$ 를 냈다. DDIM 으로 $t = 2$ 로 건너뛴 값은?

<details><summary>답 보기</summary>

① schedule 표에서 $\sqrt{\bar\alpha_5} = 0.39$, $\sqrt{1 - \bar\alpha_5} = 0.92$, $\sqrt{\bar\alpha_2} = 0.85$, $\sqrt{1 - \bar\alpha_2} = 0.53$
을 읽는다.

② 원래 값을 어림한다.

$$
\hat x_0 = (1.0 - 0.92 \times 0.5) \div 0.39 = 0.54 \div 0.39 \approx 1.38
$$

③ $t = 2$ 의 식에 같은 $\epsilon_\theta$ 와 함께 넣는다.

$$
x_2 = 0.85 \times 1.38 + 0.53 \times 0.5 \approx 1.17 + 0.27 = 1.44
$$

답: 약 1.44.

</details>

## Diffusion Policy

로봇의 policy 는 카메라 이미지나 관절 위치 같은 관측을 받아 action 을 내는 network 다. 서론의 상자 장면처럼 같은 관측에 정답 action 이
여럿인 시연은 흔하고, 왼쪽과 오른쪽으로 비켜 간 시연이 섞이면 정답 action 의 분포에 봉우리가 둘 생긴다. 이렇게 봉우리가 여럿인 분포를
**multimodal** 이라고 한다. 2023 년 Chi 등의 Diffusion Policy 는 DDPM 이 만드는 대상을 action 으로 바꾸고 network 에 관측을 함께 넣어, 두
길의 평균인 가운데 길 대신 시연에 있던 길 하나를 만든다.

![상자를 비켜 가는 시연이 왼쪽과 오른쪽 반반일 때 MSE policy 는 가운데 길로 상자에 부딪히고, Diffusion Policy 는 한쪽 길을 뽑는다](figures/policy-paths.svg)

Diffusion Policy 의 $x_0$ 는 그림이 아니라, 앞으로 할 action 여러 개를 시간 순서대로 묶은 **action chunk** 다. 책상 위의 T 자 블록을 밀어 정해진
자리에 놓는 PushT task 에서는 손끝이 갈 평면 위치 $(x, y)$ 16 개를 이어 붙인 숫자 32 개가 $x_0$ 가 된다. network 는 noise 가 섞인 action
chunk 와 $t$ 에 더해 최근 관측 2 개를 받아 섞인 noise 를 맞힌다. 조건을 함께 넣는 이 방식은 [VAE](../vae/README.md) 에서 본 CVAE 의
decoder 와 같다. 논문의 PushT 실험에서 손끝은 같은 장면에서 블록의 왼쪽으로도 오른쪽으로도 돌아갈 수 있었고, Diffusion Policy 는 두 방식을
모두 배운 뒤 한 번의 실행 안에서는 한쪽만 끝까지 따랐다.

![Diffusion Policy 의 PushT 설정. 관측 2 개를 조건으로 noise 16 × 2 를 걷어 내 손끝 목표 16 개짜리 action chunk 를 만들고, 앞 8 개만 실행한 뒤 다시 만든다](figures/dp-pipeline.svg)

simulation 실험에서는 $T = 100$ 으로 학습한 network 로 sampling 도 100 step 을 다 밟았고, 실물 실험에서는 [DDIM](#ddim) 으로 step 을
줄였다. 실물 PushT 의 policy 는 0.1 초마다 손끝 목표를 냈고, 그 사이를 보간한 명령은 1초에 125 번씩 UR5 팔로 갔다.

LeRobot 에 올라 있는 PushT 용 Diffusion Policy(`lerobot/diffusion_pusht`)는 `horizon: 16`, `n_action_steps: 8`, `n_obs_steps: 2`,
`num_train_timesteps: 100` 으로 논문의 simulation 설정과 같다. sampling 쪽은 `noise_scheduler_type: DDPM` 에 `num_inference_steps` 가
비어 있어 학습과 같은 100 step 을 밟는다. 논문의 실물 로봇처럼 sampling 시간을 줄이려면 `noise_scheduler_type: DDIM` 과
`num_inference_steps: 10` 으로 바꾸면 되고, 다시 학습할 필요는 없다. DDIM 으로 줄여도 action 을 만들 때마다 network 를 10 ~ 16 번은
써야 한다.

**확인 문제.** 컵을 집는 시연의 절반은 손잡이를 왼쪽에서 잡았고 나머지 절반은 오른쪽에서 잡았다. MSE policy 와 Diffusion Policy 는 각각
어떤 손 위치를 낼 가능성이 클까?

<details><summary>답 보기</summary>

① MSE policy 는 같은 관측에 대한 정답들의 평균을 내므로, 왼쪽에서 잡는 손 위치와 오른쪽에서 잡는 손 위치의 가운데를 내므로 시연에 없던
위치가 나온다.

② Diffusion Policy 는 sampling 할 때마다 왼쪽에서 잡는 위치나 오른쪽에서 잡는 위치 가운데 하나에 가까운 값을 낸다.

답: MSE policy 는 두 위치의 가운데를 내고, Diffusion Policy 는 둘 가운데 한 위치를 낸다.

</details>

## 정리

- 한 번에 만드는 생성은 정답이 갈라진 데이터에서 평균을 낸다. DDPM 은 데이터를 noise 로 망가뜨리는 과정을 잘게 나눠 정해 두고, 그 step
  하나를 되돌리는 법을 network 에게 배우게 해서 평균을 피한다
- forward process 는 원래 값의 몫을 줄이며 noise 를 섞어 끝 상태를 늘 $\mathcal N(0, 1)$ 로 만든다.
  $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ 으로 어느 step 의 문제든 한 번에 만들고, 정답은 직접 뽑은 noise $\epsilon$ 이다
- sampling 은 $\mathcal N(0, 1)$ 에서 출발해 맞힌 noise 의 몫을 빼고 새 noise 를 더하는 reverse step 을 $T$ 번 밟는다. 평균은 각 step 의 작은
  이동에만 쓰여서, $\pm 2$ 데이터로 만든 sample 4000 개 가운데 봉우리 사이($-1.5$ ~ 1.5)에서 끝난 것은 하나도 없었다
- 대가는 sample 하나에 network 를 $T$ 번 쓰는 시간이다. DDIM 은 같은 network 로 건너뛰어 20 ~ 100 step 에 1000 step 과 비슷한 품질을 내고,
  Diffusion Policy 는 실물 로봇에서 10 step 만 밟는다(논문 부록에는 16 step 으로 적혀 있다)

noise 에서 데이터로 가는 경로를 처음부터 곧게 학습해 몇 step 만에 sample 을 만드는 방법은 [Flow Matching](../flow-matching/README.md) 에서
다루고, 관측을 조건으로 넣는 network 와 action chunk 를 실행하는 방식은 [Diffusion Policy](../diffusion-policy/README.md) 에서 다룬다.

## 참고 자료

- Jonathan Ho, Ajay Jain, Pieter Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020) — DDPM 의 원
  논문. Algorithm 1(학습)과 Algorithm 2(sampling)가 이 강의의 학습 순서와 sampling 식이고, 4 절에 $T = 1000$ 과 linear schedule 이 있다
- Jiaming Song, Chenlin Meng, Stefano Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2020) — 같은 network 로
  step 을 건너뛰는 DDIM. DDPM 의 sampling 시간(서론)과 step 수별 FID(표 1)를 본다
- Alex Nichol, Prafulla Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (2021) — 3.2 절의
  cosine schedule 과 linear schedule 의 $\bar\alpha_t$ 비교 그림
- Cheng Chi 등, [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) (2023) — DDPM 으로 action
  chunk 를 만드는 policy. DDIM 10 step 과 0.1 초(3.4 절), PushT 의 두 방식(그림 3), $T_o$·$T_a$·$T_p$ 와 실물 실험의 sampling step 수(부록 표 7)를
  본다
- LeRobot, [configuration_diffusion.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py) —
  Diffusion Policy 설정의 기본값과 각 parameter 의 설명. pretrained PushT 모델의 설정은
  [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht) 에 있다
- Jascha Sohl-Dickstein 등, [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (2015) — 데이터를
  noise 로 흩뜨렸다가 되돌리는 생성 모델을 처음 제안한 논문. 2.2 절에서 작은 step 의 되돌리기가 같은 꼴이 되는 근거(Feller, 1949)를 든다
- Calvin Luo, [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) — VAE 의 ELBO 에서 출발해 DDPM 의 loss 와
  reverse step 의 계수를 한 줄씩 유도하는 tutorial
- Lilian Weng, [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — forward·reverse 식의 유도와 DDIM
  등 뒤이은 방법을 한 글에 정리한 블로그
- Hugging Face, [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — DDPM 을 U-Net 과 time embedding 까지 구현하며
  해설한 글
