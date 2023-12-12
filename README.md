# essay_writing_evaluation_kor
Shallow-learning with essay writing evaluation data (KOR)

## 개발 환경

- Google Colab
- IPython
- Pandas, Numpy, Cupy
- KoNLPy, WordCloud
- 밑바닥부터 시작하는 딥러닝 2 코드 활용 ([GitHub](https://github.com/WegraLee/deep-learning-from-scratch-2))

## 데이터셋 소개

- 다양한 학년군의 에세이 및 에세이 평가 점수로 구성된 데이터
- [AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=545)

## 데이터 상관관계 분석

### 학생 학년 별 평균 점수
![학생 학생 별 평균 점수](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/97efc5f9-ac4a-487a-980f-e4b4355feff1)

### 학생이 읽은 책의 양 별 평균 점수
![학생이 읽은 책의 양 별 평균 점수](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/e5af75ca-83a6-4c21-b96b-edbc6094a8b5)

### 에세이 길이 별 평균 점수
![에세이 길이 별 평균 점수](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/4650a523-f64f-4737-9778-f88a95a7e2d0)

## 데이터 전처리

### JSON to DataFrame
```python
question = []
answer = []
filelist = os.listdir(path)
print(filelist)
for file in filelist:
  if file.endswith('.json'):
    filepath = path + file
    with open(filepath, 'r') as f:
      buf = json.load(f)
      paragraph_txt = buf['paragraph'][0]['paragraph_txt'].replace('#@문장구분#', '')
      paragraph_score = buf['score']['paragraph_score'][0]['paragraph_scoreT_avg']
      #print(paragraph_score)

      essay_score = buf['score']['essay_scoreT_avg']
      #print(essay_score)

      score_txt = "_"+str(round(paragraph_score+essay_score))
      question.append(paragraph_txt)
      answer.append(score_txt)
```

### 학습 데이터 길이 고정
```python
def set_word_fixed_len(df):
  q_len = df['question'].apply(len).max()
a_len = df['answer'].apply(len).max()

  def question_pad_string(s):
    return s.ljust(q_len+5)

  def answer_pad_string(s):
    return s.ljust(a_len+2)

  df['question'] = df['question'].apply(question_pad_string)
  df['answer'] = df['answer'].apply(answer_pad_string)
  df['q_length'] = df['question'].apply(len)
  df['a_length'] = df['answer'].apply(len)
```

### GPU 에러 해결

#### `deep_learning_from_scratch_2/common/np.py`
- np.add.at 주석 추가
```python
# coding: utf-8
from common.config import GPU
if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    #np.add.at = np.scatter_add
```

#### `deep_learning_from_scratch_2/common/layers.py`
- Embedding 클래스의 backward 메서드 수정 
```python
def backward(self, dout):
  dW, = self.grads
  dW[...] = 0
  if GPU:
    import cupyx
    cupyx.scatter_add(dW, self.idx, dout)
  else:
    np.add.at(dW, self.idx, dout)
  return None

```

#### `deep_learning_from_scratch_2/common/time_layers.py`
- TimeSoftmaxWithLoss 클래스의 forward 메서드 수정
```python
def forward(self, xs, ts):
    N, T, V = xs.shape
    if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
        ts = ts.argmax(axis=2)
    mask = (ts != self.ignore_label)
    mask = np.array(mask)
    #print(type(mask), "mask")
    #print(mask)
```

### 주요 단어 추출
```python
import konlpy
from konlpy.tag import Okt
from konlpy.utils import pprint

def list2seq(lst):
return ' '.join(lst)

def seq2list(seq):
  # seq example) "인생 가장 경험 미국 심장 강타 사건 피해자 마지막 순간 메시지 사업 회사"
return seq.split(' ')

def get_nouns_seq_with_no_dup(text):
  okt = Okt()
  nouns_list = okt.nouns(text)
  nouns_list = [noun for noun in nouns_list if len(noun) > 1] # 1글자는 제외
  nouns_list = list(set(nouns_list))  # 중복제거
return list2seq(nouns_list)

def get_nouns_seq_with_dup(text):
  okt = Okt()
  nouns_list = okt.nouns(text)
  nouns_list = [noun for noun in nouns_list if len(noun) > 1] # 1글자는 제외
  return list2seq(nouns_list)
```

### 단어 빈도 수 확인
`생각 여러분 미래 사람과 모두 우리 가요 노력 각각 상상 소통 가치관 작성 의사 가지 사람 뒷모습 모습 위해`라는 에세이 주제문의 주요 단어가 입력되었을 때의 에세이 주요 단어 시각화 결과
![에세이 단어 빈도 수 시각화](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/7b708762-392b-48b7-a261-5d549f6da189)

### 학습데이터 로드 (DataFrame to Training set)
```python
def load_data_from_dataframe(df, seed=1984):
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    #print(questions[0:20])
    #print(answers[0:20])
    # 어휘 사전 생성 (기존 코드에서 _update_vocab 함수 구현 필요)
    for q, a in zip(questions, answers):
        _update_vocab(q)
        _update_vocab(a)

    # 넘파이 배열 생성
    x = numpy.zeros((len(questions), len(questions[0])), dtype=numpy.int)
    t = numpy.zeros((len(answers), len(answers[0])), dtype=numpy.int)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # 뒤섞기
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)

    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 검증 데이터셋으로 10% 할당
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]
    return (x_train, t_train), (x_test, t_test)
```

## 에세이 평가 점수 예측

- 평가하려는 에세이를 입력했을 때, 해당 에세이의 점수를 예측하는 모델
- `Predict_Score_Deep.ipynb` 코드 참고

### 1) Essay to Score

#### 개요

에세이 전체 글을 학습하여 평가 점수를 예측하는 모델이다.

#### 학습 결과

![Essay to Score 모델 점수 예측 결과](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/4169ded8-f8cd-4067-ac78-5ef6132f657e)

![Essay to Score 모델 평가 그래프](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/3f3eb9d3-a4d6-4eda-b8d0-2fa2c6f043b2)

### 2) Essay word to Score

#### 개요

에세이 전체 글 중, 명사 단어만 추출 및 학습하여 해당 에세이의 평가 점수를 예측하는 모델이다. 

#### 학습 결과

![Essay word to Score 모델 점수 예측 결과](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/690b32fa-5b9a-4545-a8b0-83660c01b17f)

![Essay word to Score 모델 평가 그래프](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/2146c908-19de-4795-8236-53b683f4fc25)


### 3) Ridge Regression with ML

- `Predict_Score_Machine.ipynb` 코드 참고

#### 개요

딥러닝 모델과 성능 비교를 위해 머신러닝 모델 생성했다. Ridge Regression을 이용하였으며, 이용한 피처는 다음과 같다.

| 데이터 속성 | 설명 |
| --- | --- |
| score | 문단 점수 + 에세이 점수 |
| student_educated | 학생의 논술 사교육 유무 |
| student_grade | 학생 학년 |
| student_reading | 학생이 일주일 간 읽은 책의 양 |
| essay_prompt_len | 에세이 프롬프트 길이 |
| essay_level | 에세이 난이도 |
| essay_len | 에세이 길이 |

#### 학습 결과
- MSE = 9.87
![머신러닝 모델 예측값 실제정답 비교 그래프](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/9426ce22-a9e2-4c3d-9e90-b92813652839)

## 에세이 자동 생성

- 에세이의 주제를 입력하면 해당 주제에 맞게 에세이를 생성하는 모델
- `Generate_Essay.ipynb` 코드 참고

### 1) Essay prompt to Essay

#### 개요

에세이의 주제문을 입력받아 학습하여 에세이를 출력하는 모델이다.

#### 학습 결과
![Essay prompt to Essay 학습 결과](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/97a7f7bd-c2b0-4cc6-bc95-d8aa3196e57e)

### 2) Essay prompt word to Essay word

#### 개요

에세이의 주제문 중 빈도 수가 높은 단어를 추출 및 학습하여 에세이를 구성하는 주요 단어를 생성하는 모델이다.

#### 학습 결과
![Essay prompt word to Essay word 학습 결과](https://github.com/leehahoon/essay_writing_evaluation_kor/assets/15906121/82d7a272-53ea-4fa7-ab40-7dbc98fae6ac)

#### 학습 결과 요약
| 에세이 주제 단어 | 에세이 단어 | 모델 생성 단어 |
| --- | --- | --- |
| 매일 매우 호의 사람 기분 경험 만약 본인 명언 **부모님** | **가족** 사랑 **사람** 남편 인생 존재 가장 마지막 순간 메시지 | **친구** 생각 서로 **사람** **가족** 학교 |
| 상상 **미래** 사람 여러분 우리 모두 사람과 소통 의사 생각 | 신발 **생각** 위해 **공부** 디자인 브랜드 지금 디자이너 노력 사람 | **생각** **미래** 모습 상상 **대학교** 때문 **대학** 졸업 |
| **미래** **도시** 모습 생각 작성 기술 발전 자동차 하늘 사람 | **생각** **미래** 환경 모습 사람 편의 로봇 친환경 파괴 | **미래** 도시 **발전** **생각** 지금 우리 정말 이름 |
| **바다** **생물** 모험 수도 지구 표면 차지 무수 종류 정말 | 만두 **쭈꾸미** **상어** 거북이 옛날 소식 쭈꿈 그때 친구 은서 | **바다** 생물 **상어** **물고기** 때문 일주일 동안 모습 만약 |
