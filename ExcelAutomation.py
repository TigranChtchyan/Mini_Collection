import openpyxl
import math
from tabulate import tabulate
from PIL import Image, ImageDraw, ImageFont, ImageMath
import numpy as np

def krk(sheet, i, j):
	i1=sheet['D'+str(i)].value
	i2=sheet['E'+str(i)].value
	j1=sheet['D'+str(j)].value
	j2=sheet['E'+str(j)].value
	if(i1==j1 and i2==j2):
		return True
	else:
		return False
def kgm(d):
	return math.pi*d*d*0.01/4*0.785
def hatk(sheet,a,b):
	l=[]
	sumnl=[]
	for i in range(a,b+1):
		b=True
		for j in range(a,i):
			if(krk(sheet, i, j)==True):
				b=False
				sumnl[j-a]+=sheet['H'+str(i)].value
				break
		if(b==True):
			l.append([sheet['D'+str(i)].value, sheet['E'+str(i)].value])
			sumnl.append(sheet['H'+str(i)].value)
	for k in range(a,a+len(l)):
		sheet['I'+str(k)]=l[k-a][0]
		sheet['J'+str(k)]=l[k-a][1]
		sheet['K'+str(k)]=sumnl[k-a]
		sheet['L'+str(k)]=kgm(l[k-a][0])
		sheet['M'+str(k)]=sheet['L'+str(k)].value*sheet['K'+str(k)].value

def im(sheet,l,togh):
	if(len(l)%2==0):
		shape=(int(len(l)//2),2)
	else:
		l.append('-')
		shape =(int(len(l)/2),2)
	data=np.array(l)
	data.shape=shape
	print(data.shape)
	im=tabulate(data, tablefmt="orgtbl")
	sheet['C'+str(togh)]=im

wb=openpyxl.load_workbook('chkrknvogh.xlsx')
sheet=wb['Sheet']

print('Ete uzum eq C syunum nor ban avelacnel, seghmeq 1')
print('Isk ete petq e anel hatkavorum, seghmeq 2')
#t=input()
t=2
if(t==1):
	print('Vor toghum eq uzum avelacnel')
	#togh=input()
	togh=7
	print('qani koghm uni obyekty?')
	#count=input()
	count=5
	print('greq koghmeri erkarutyunnery')
	x=[]
	for i in range(count):
		#x.append([input()])
		x.append(500)
	sum=0
	for i in range(count):
		sum+=x[i]
	im(sheet, x, togh)
	sheet['F'+str(togh)]=sum
else:
	print('nermuceq skzbnakan toghy')
	#a=input()
	a=3
	print('nermuceq verjnakan toghy')
	#b=input()
	b=6
	hatk(sheet,a,b)
wb.save('chkrknvogh.xlsx')