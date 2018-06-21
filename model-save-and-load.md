# Save and Load

모델을 저장할 때, 아래 형태로 예제가 많이 나와있다.
```python
torch.save(model.state_dict(), FILENAME)  # for saving
model.load_state_dict(torch.load(FILENAME))  # for loading
```
model은 저장 및 불러오기를 하고자 하는 모델 클래스(nn.Module type)이고, FILENAME은 파일 이름이다.

그런데 위의 코드를 사용해서 training을 이어서 했더니 loss가 매번 continue를 할 때마다 크게 튀었다.
이어서 학습되지 않는 것으로 보였다.

pytorch save/load 관련 문서를 찾다가 포럼에서 [이 예제](https://discuss.pytorch.org/t/how-to-resume-training/8583/2)를 찾았는데 핵심은 optimizer의 파라미터도 저장해야 한다는 것이다.
아래와 같이 실행해야 training이 이어서 진행된다.

```python
# for saving
torch.save({
    'state': model.state_dict(),
    'optim': optimizer.state_dict(),
}, FILENAME)

# for loading
checkpoint = torch.load(FILENAME)
model.load_state_dict(ckpt['state'])
optimizer.load_state_dict(ckpt['optim'])
```
