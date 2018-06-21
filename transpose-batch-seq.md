# RNN 학습 시 (batch, seq, features) 입력을 (seq, batch, features)로 변환하기

nn.LSTM이나 nn.GRU를 적용하고자 할 때 [기본 입력은 (seq, batch, features)](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM)이다. 

하지만 보통 학습 시 batch가 dim 0인 경우가 많다.
보통은 (batch, seq, features)의 형태의 입력을 처리하기 위해 nn.LSTM이나 nn.GRU가 가지고 있는 batch_first 옵션을 사용할 것이다.
단일 GPU를 사용할 경우, 이는 잘 적용된다.

단, nn.parallel.data_parallel()과 같이 multi-GPU를 활용할 때 문제가 발생한다.
nn.LSTM이나 nn.GRU의 또 다른 입력인 hidden state는 batch가 dim 1에 고정되어 있기 때문이다.

따라서 multi-GPU 학습을 위해서는 입력 데이터를 (seq, batch, features)로 변경해주고 nn.parallel.data_parallel(MODEL, INPUT, dim=1) 형태로 실행해야 한다.

(batch, seq, features) => (seq, batch, features)를 위해 인터넷에서 검색을 해보니 다음 두 가지 중 하나를 사용하고 있었다.

1. x.transpose(0, 1)
2. x.permute(1, 0, 2)

처음에 2를 먼저 보고 사용하다가 1를 보게 되었고 두 방식 모두 잘 작동했다.
```python
a = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]], dtype=torch.int)

tensor([[[  1,   2,   3],
         [  4,   5,   6],
         [  7,   8,   9],
         [ 10,  11,  12]],

        [[ 13,  14,  15],
         [ 16,  17,  18],
         [ 19,  20,  21],
         [ 22,  23,  24]]], dtype=torch.int32)

a.permute(1, 0, 2)
tensor([[[  1,   2,   3],
         [ 13,  14,  15]],

        [[  4,   5,   6],
         [ 16,  17,  18]],

        [[  7,   8,   9],
         [ 19,  20,  21]],

        [[ 10,  11,  12],
         [ 22,  23,  24]]], dtype=torch.int32)

a.transpose(0, 1)
tensor([[[  1,   2,   3],
         [ 13,  14,  15]],

        [[  4,   5,   6],
         [ 16,  17,  18]],

        [[  7,   8,   9],
         [ 19,  20,  21]],

        [[ 10,  11,  12],
         [ 22,  23,  24]]], dtype=torch.int32)
```

그렇다면 더 빠르게 동작하는 것을 사용해야 할 터.
```python
%timeit(a.permute(1, 0, 2))
1.29 µs ± 64.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

%timeit(a.transpose(0, 1))
983 ns ± 34.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

transpose()를 사용하자.
