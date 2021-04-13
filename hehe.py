# Enter your code here. Read input from STDIN. Print output to STDOUT
import fileinput

# list <=> stack, only use pop, append
class Myqueue:
    def __init__(self):
        self.head = []
        self.tail = []
    def enqueue(self,num):
        self.tail.append(num)
    def dequeue(self):
        if self.head:
            self.head.pop()
        else:
            if self.tail:
                self.tailToHead().pop()
            else:
                pass
                


    def print(self):
        if self.head:
            var =self.head.pop()
            self.head.append(var)
            return var
        else:
            if self.tail:
                var = self.tailToHead().pop()
                self.head.append(var)
                return var 
            else:
                pass

    def tailToHead(self):
        self.head = [self.tail.pop() for _ in range(len(self.tail))]
        return self.head


k = Myqueue()
k.enqueue(11)
k.dequeue()
k.enqueue(42)


print(k.print())
k,enqueue(28)
print(k.print())


