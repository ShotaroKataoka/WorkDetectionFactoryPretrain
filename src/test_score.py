import numpy as np
import statistics


# time, pred
result = np.loadtxt('result/result_0915.csv', delimiter=',', dtype=float).astype(int)
# time, pred, target
ans = np.loadtxt('result/result_0915.csv', delimiter=',', dtype=float).astype(int)

# 秒ごとにまとめる
result_tmp = {}
for res, a in zip(result, ans):
  time = str(res[0])
  pred = res[1]+1
  target = a[2]+1
  try:
    result_tmp[time]['pred'] += [pred]
  except:
    result_tmp[time] = {'target': target, 'pred': [pred]}

# 秒ごとに最頻値
for time in result_tmp.keys():
  result_tmp[time]['pred'] = statistics.mode(result_tmp[time]['pred'])

# 混同行列を用意
conf = {}
for row in range(1, 13):
  for col in range(1, 13):
    conf[f'{row}/{col}'] = 0

# 混合行列を作成
for time in result_tmp.keys():
  res = result_tmp[time]
  pred = res['pred']
  target = res['target']
  conf[f'{target}/{pred}'] += 1

print(conf)


# 再現率/適合率
tp = {}
recall = {}
precision = {}
for t in range(1, 13):
  for p in range(1, 13):
    v = conf[f'{t}/{p}']
    if t==p:
      tp[t] = v
    try:
      recall[t] += v
    except:
      recall[t] = v
    try:
      precision[p] += v
    except:
      precision[p] = v

f1 = {}
for key in tp.keys():
  precision[key] = tp[key] / (precision[key] + 0.0000001)
  recall[key] = tp[key] / (recall[key] + 0.0000001)
  f1[key] = 2*recall[key]*precision[key] / (recall[key] + precision[key] + 0.0000001)

for key in sorted(precision.keys()):
  print(f'precision[{key}]:', precision[key])
print()
for key in sorted(recall.keys()):
  print(f'recall[{key}]:', recall[key])
print()
f = 0
for key in sorted(f1.keys()):
  print(f'f1[{key}]:', f1[key])
  f += f1[key]

print()
print('class mean f1', f/12)

