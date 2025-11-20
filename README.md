# 📝 LSTM 기반 영화 리뷰 감성 분석 (Sentiment Analysis)

## 🚀 프로젝트 개요

이 프로젝트는 $\text{IMDb}$ 영화 리뷰 데이터셋을 사용하여 \*\*순환 신경망(RNN)\*\*의 한 종류인 **$\text{LSTM}$ (Long Short-Term Memory)** 모델을 구축하고 훈련시켜, 입력된 영화 리뷰 텍스트가 **긍정(1)인지 부정(0)인지**를 예측하는 **이진 분류(Binary Classification)** 태스크를 수행합니다.

이 프로젝트는 텍스트 처리의 기본인 **임베딩(Embedding)** 개념과 **시퀀스 데이터 처리**를 위한 $\text{LSTM}$ 모델 구조를 이해하는 것을 목표로 합니다.

-----

## 🛠️ 사용 기술 및 모델 구조

### 1\. 주요 기술

  * **언어:** Python
  * **프레임워크:** TensorFlow, Keras
  * **데이터셋:** $\text{IMDb}$ 영화 리뷰 데이터셋

### 2\. 모델 아키텍처 (Model Architecture)

| 레이어 (Layer) | 역할 | 설명 |
| :--- | :--- | :--- |
| **$\text{Embedding}$** | **단어 벡터화** | 10,000개의 단어를 입력받아 각 단어를 **128차원의 밀집 벡터**로 변환합니다. 이는 단어의 의미적/문법적 정보를 학습하는 핵심 단계입니다. |
| **$\text{LSTM(128)}$** | **시퀀스 학습** | $\text{RNN}$의 **장기 의존성 문제**를 해결하는 핵심 레이어입니다. 입력 시퀀스의 **맥락과 순서**를 128개의 은닉 상태 노드를 통해 학습합니다. |
| **$\text{Dense(1)}$** | **최종 분류** | $\text{LSTM}$의 출력을 받아 **Sigmoid 활성화 함수**를 통해 최종적으로 긍정(1) 또는 부정(0)에 대한 **확률**을 출력합니다. |

-----

## 💻 실행 단계별 상세 설명

### 1단계: 환경 설정 및 데이터 로드

$\text{TensorFlow}$의 `imdb` 데이터셋을 로드합니다.

  * `num_words=10000` 설정으로 가장 자주 등장하는 **10,000개의 단어**만 사용하여 분석에 포함합니다.

<!-- end list -->

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# print(x_train.shape) # (25000,)
```

### 2단계: 데이터 전처리 (Pre-processing)

텍스트 데이터를 신경망 모델에 입력하기 위해 표준화합니다.

1.  **토큰화 ($\rightarrow$ 정수 인코딩):** (데이터셋에 의해 이미 완료) 문장을 단어로 쪼개고 각 단어에 고유한 정수(숫자)를 부여하여 텍스트를 정수 시퀀스로 변환합니다.
2.  **패딩 ($\text{Padding}$):** 모든 리뷰의 길이가 다르므로, `pad_sequences`를 사용하여 길이를 **500**으로 통일합니다. 부족한 길이는 특수 숫자(0)로 채워집니다.

<!-- end list -->

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_review_length = 500
# x_train = pad_sequences(x_train, maxlen=max_review_length)
# print(x_train.shape) # 결과: (25000, 500)
```

### 3단계: $\text{LSTM}$ 모델 구축

$\text{Embedding}$ 레이어를 시작으로 $\text{LSTM}$ 레이어와 $\text{Dense}$ 출력 레이어를 순차적으로 연결하여 모델을 구성합니다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    # input_dim=10000 (단어 개수), output_dim=128 (임베딩 차원), input_length=500 (시퀀스 길이)
    Embedding(input_dim=10000, output_dim=128, input_length=max_review_length),
    LSTM(128),
    # 최종 출력: 1개 노드, 시그모이드(Sigmoid) 활성화 함수로 0~1 사이의 확률 출력
    Dense(1, activation='sigmoid')
])
```

### 4단계: 모델 훈련 및 평가

모델을 컴파일하고 훈련시킨 후, 테스트 데이터로 성능을 평가합니다.

  * **컴파일:** `optimizer='adam'`을 사용하고, 이진 분류에 적합한 \*\*`loss='binary_crossentropy'`\*\*를 손실 함수로 사용합니다.
  * **훈련:** `epochs=10`, `batch_size=64`, 검증 데이터(`validation_split=0.2`)를 분리하여 학습합니다.

| 메트릭 (Metric) | 결과 값 |
| :--- | :--- |
| **Test Loss** | $0.7181$ |
| **Test Accuracy** | $0.8490$ |

테스트 데이터에 대한 최종 \*\*정확도(Accuracy)\*\*는 \*\*84.90%\*\*로, 모델이 영화 리뷰의 감성을 잘 분류하고 있음을 보여줍니다.

### 5단계: 예측 결과 확인

테스트 샘플을 입력하여 예측을 시도합니다.

```python
# sample_review = x_test[0].reshape(1, max_review_length)
# prediction = model.predict(sample_review)
# print(f"예측 확률: {prediction[0][0]: .4f}") # 결과: 0.0161

# 예측 결과: 부정 (Negative)
```

예측 확률 $0.0161$은 **0.5 미만**이므로, 이 리뷰를 \*\*부정(Negative)\*\*으로 예측했습니다.

-----

## 📚 학습을 위한 추가 참고 사항

본 $\text{LSTM}$ 모델의 작동 원리를 이해하는 데 중요한 두 가지 활성화 함수의 역할은 다음과 같습니다.

  * **$\text{Sigmoid}$ 함수 ($\sigma$):** $\text{LSTM}$ 내부에서 \*\*게이트(Gate)\*\*의 제어 역할($0 \sim 1$)을 하며, 최종 $\text{Dense}$ 레이어에서 **이진 분류의 확률**을 출력하는 데 사용됩니다. (예: 긍정일 확률 0.8)
  * **$\tanh$ 함수:** $\text{LSTM}$ 내부에서 **후보 셀 상태 ($\tilde{C}_t$)** 및 최종 **은닉 상태($h_t$)** 계산 시 **값의 크기($-1 \sim 1$)를 안정적으로 정규화**하고 정보의 방향을 표현하는 데 사용됩니다.
