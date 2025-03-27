import statistics
import random
def rnd():
    while True:
        x = random.uniform(-2, 2)
        if abs(x) > 1:
            return x
x0 = 0.5
y0 = 0.5
ans = [ ]
sc = 0.15
for i in range(2):
    for j in range(3):
        ans.append((i * sc, j * sc))
ans = ans[:5]
xc = statistics.fmean(x[0] for x in ans)
yc = statistics.fmean(x[1] for x in ans)

ans = [ (x - xc + x0, y - yc + y0) for x, y in ans ]

fmt = "windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=%g ypos=%g bBlockAng=1 angvelmax=%g freq=0.25"
for x, y in ans:
    print(fmt % (x, y, rnd()))
# print(ans)
