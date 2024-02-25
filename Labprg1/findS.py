import csv

num_attributes=6
a=[]
print("The given training data set is\n")

with open('D:/enjoysport.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        a.append(row)
        print(row)

print("The initial value of hypothesis\n")
hypothesis=['0']*num_attributes

for j in range(num_attributes):
    hypothesis[j]=a[0][j]

print("FindS: Finding a maximally specific hypothesis:\n ")
for i in range(0,len(a)):
    if a[i][num_attributes]=="yes":
        for j in range(num_attributes):
            if a[i][j]!=hypothesis[j]:
                hypothesis[j]='?'
            else:
                hypothesis[j]=a[i][j]
                
    print("For the training instance:{0} , the hypothesis is: ".format(i),hypothesis)
print("The Maximally specific hypothesis is: \n")
print(hypothesis)

        