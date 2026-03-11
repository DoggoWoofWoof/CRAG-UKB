import csv

with open('benchmark_400.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = [lines[0]]
for l in lines[1:]:
    p = l.strip('\n').split(',')
    if len(p) > 4:
        q = ','.join(p[1:-2])
        out.append(f'{p[0]},"{q}",{p[-2]},{p[-1]}\n')
    else:
        out.append(l)

with open('benchmark_400.csv', 'w', encoding='utf-8') as f:
    f.writelines(out)
