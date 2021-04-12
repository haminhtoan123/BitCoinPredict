#READ Input
input = open("input.txt")
output = open("output.txt", "w")

n = int(input.readline())

money_arr = list(map(int, input.read().splitlines()))

total = money_arr.pop()
print(total)
print(money_arr)

i=0
output_var =0
while (i<len(money_arr)):
    if total - money_arr[i] in money_arr:
        money_arr =list(filter(lambda a: a != total - money_arr[i] and a !=money_arr[i] , money_arr))
        #money_arr.remove(total - money_arr[i])
        #money_arr.remove(money_arr[i])
        output_var+=1
    else:
        i+=1

output.write(str(output_var))
input.close()
output.close()
