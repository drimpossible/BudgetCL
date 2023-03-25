leafs, children = {}, {}

f1 = open('is-a-relation-synsets.txt','r')
lines = f1.readlines()

f2 = open('cls.txt','r')
lines_imgnet = f2.readlines()

for line in lines:
    parent, child = line.strip().split(' ')
    if parent not in children: children[parent] = [child]
    else: children[parent].extend([child])
    
    leafs[parent] = 0
    if child not in leafs: leafs[child] = 1
    
    
def dfs(pline):
    all_children = []    
    if pline in children:
        all_children.extend(children[pline])
        for child in children[pline]: 
            all_children.extend(dfs(pline=child))
    return all_children

problematic = 0
numparent = 0
numchild = 0
not_present = 0

for line in lines_imgnet:
    if line.strip() not in leafs: 
        not_present += 1
        print(line)
    elif leafs[line.strip()] == 1: numchild += 1
    elif leafs[line.strip()] == 0: 
        numparent += 1
        all_children = dfs(pline=line.strip())
        #print(all_children)
        flag = 0
        for child in all_children:
            if child in lines_imgnet: 
                if not flag:
                    flag = 1
                    problematic +=1
                print(line.strip(), child)

print(problematic, numparent, numchild, not_present)
print(len(lines_imgnet), numparent + numchild + not_present)
