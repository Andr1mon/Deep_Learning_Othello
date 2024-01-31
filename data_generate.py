import os
import random
total = os.listdir('AI_dataset')
n = len(total)
random.shuffle(total)
f = open(f'dev_AI.txt', 'w', encoding='utf-8')
for i in range(int(n/3)):
    f.write(f'{total[i]}\n')
f.close()
f = open(f'train_AI.txt', 'w', encoding='utf-8')
for i in range(int(n/3), n):
    f.write(f'{total[i]}\n')
f.close()