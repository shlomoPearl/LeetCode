import math
from time import sleep

def isValid(s: str) -> bool:
    q = []
    i = 0
    open_p = {'(':0, '[':1, '{':2}
    close_p = {')':0, ']':1, '}':2}
    while(i < len(s)):
        if(s[i] in open_p):
            q.append(s[i])
        elif(s[i]in close_p):
            if(not q):
                return False
            p = q.pop()
            if((open_p.get(p) - close_p.get(s[i], -1)) != 0):
                return False
        i += 1
    if(q):
        return False
    return True
      

def simplifyPath(path: str) -> str:
    dir_list = path.split('/')
    simplify = []
    for dir in dir_list:
        if dir == '' or dir == '.':
            continue
        if dir == '..':
            if simplify:
                simplify.pop()
            # dir = ''
            continue
        simplify.append('/' + dir)
    if not simplify:
        return '/'
    simplify_str = ''
    for dir in simplify:
        simplify_str += dir
    return simplify_str


def min_len_s(str_list):
    min_len = len(str_list[0])
    index_min = 0
    for i in range(1, len(str_list)):
        if(len(str_list[i]) < min_len):
            min_len = len(str_list[i])
            index_min = i
    return index_min

def longestCommonPrefix(strs: list[str]) -> str:
    count_l = 0
    i = 0
    shortest = strs[min_len_s(strs)]
    while(i < len(shortest)):
        pref = shortest[i]
        for s in strs:
            if(pref != s[i]):
                return shortest[:count_l]
        count_l+=1
        i+=1
    return shortest[:count_l]
            
def strStr(haystack: str, needle: str) -> int:
    if len(needle) > len(haystack): return -1
    for i in range(len(haystack)):
        if(needle[0] != haystack[i]): continue
        j = 1
        while(j < len(needle) and (j+i) < len(haystack)):
            if(needle[j] != haystack[j + i]): break
            j += 1
        if j == len(needle):
            return i
    return -1

def canConstruct(ransomNote: str, magazine: str) -> bool:
    letters = {}
    for l in ransomNote:
        letters[l] = letters.get(l,0) + 1
    for l in magazine:
        if (letters.get(l,0)):
            letters[l] -= 1
    for v in letters.values():
        if v: return False
    return True

def isIsomorphic(s: str, t: str) -> bool:
    letters_s = {}
    letters_t = {}
    for i in range(len(s)):
        if(letters_s.get(s[i], t[i]) != t[i] or letters_t.get(t[i], s[i]) != s[i]):
            return False
        letters_s[s[i]] = t[i]
        letters_t[t[i]] = s[i]
    return True

def maxProfit_1(prices: list[int]) -> int:
    max_profit = 0
    i = 0
    j = 1
    while(j < len(prices)):
        while(j < len(prices) and prices[j] - prices[i] <= 0):
            i+=1
            j+=1
        # max_profit = prices[j] - prices[i]
        # j+=1
        while(j < len(prices) and prices[j] - prices[i] > 0):
            if(prices[j] - prices[i] > max_profit):
                # print(prices[j] ,prices[i], i)
                max_profit = prices[j] - prices[i]
            j+=1
        i=j
        j+=1
    return max_profit

def binary_search(nums, t, offset=0):
    print(nums, offset)
    if(nums[len(nums)//2] == t or len(nums) == 1):
        # print(offset)
        return math.ceil(len(nums)/2) + offset
    elif(nums[len(nums)//2] < t):
        return binary_search(nums[len(nums)//2:], t, offset + math.ceil(len(nums)/2))
    else:
        return binary_search(nums[:len(nums)//2],t, abs(len(nums) // 2 - offset))
    # else:
    #     return offset + len(num) // 2 

def searchInsert(nums: list[int], target: int) -> int:
    if target >= nums[-1]: return len(nums)
    if target <= nums[0]: return 0
    return binary_search(nums,target)
    
def timeConversion(s):
    if 'AM' in s:
        s = s.replace('AM', '').split(':')
        if s[0] == '12':
            s[0] = '00'
        
    else:
        s = s.replace('PM', '').split(':')
        s[0] = str(int(s[0])+12)
        print(s[0])
        if s[0] == '24':
            s[0] = '12'
    return s[0]+':'+s[1]+':'+s[2]

def n_to_binary(n):
    binary = '0'*32
    count_ler = len(binary) - 1
    while(n != 0):
        binary = binary[:count_ler] + str(n % 2) + binary[count_ler + 1:]
        n //= 2
        count_ler -=1
    return binary

def binary_to_n(binary):
    n = 0
    for i in range(0, len(binary)):
        n += int(binary[i])*(2**(31-i))
    return n

def flip(binary):
    for i,bit in enumerate(binary):
        binary = binary[:i] + str(abs(int(bit)-1)) + binary[i+1:]
    return binary


def climbStairs(n: int) -> int:
    last1 = 0
    last2 = 1
    for i in range(1,n):
        tmp = last1+last2
        last1 = last2
        last2 = tmp
    return last1+last2

def reverseWords(s: str) -> str:
    s = s.split(' ')
    ans = ""
    while len(s) > 1:
        tmp = s.pop()
        if tmp == '':
            continue
        ans += tmp + ' '
    tmp = s.pop()
    if tmp == '':
        return ans[:len(ans)-1]
    return ans + tmp

def convert(s: str, numRows: int) -> str:
    zig_dict = {}
    zigzag = 0
    flag = True
    i = 0
    while i < len(s):
        zig_dict[zigzag] = zig_dict.get(zigzag,[])
        zig_dict.get((zigzag)%numRows).append(s[i])
        i+=1
        if flag:
            zigzag+=1
        else:
            zigzag-=1
        if zigzag == 0:
            flag = True
        if zigzag == numRows-1:
            flag = False
    ans = ''
    for j in range(len(zig_dict.values())):
        tmp = zig_dict.get(j)
        ans += ''.join(tmp)
    return ans

def productExceptSelf(nums: list[int]) -> list[int]:
    pre, suf = [], []
    i, j = 0, len(nums)-1
    # O(n)
    while (i < len(nums) and j >=0):
        if i == 0:
            pre.append(nums[i])
        else:
            pre.append(pre[i-1]*nums[i])
        if j == len(nums)-1:
            suf.insert(0, nums[j])
        else:
            suf.insert(0, suf[0]*nums[j])
        i+=1
        j-=1
    ans = []
    # O(n)
    for i in range(len(nums)):
        if i == 0:
            ans.append(suf[1])
        elif i == len(nums)-1:
            ans.append(pre[-2])
        else:
            ans.append(pre[i-1]*suf[i+1])
    return ans
    
def canJump(nums: list[int]) -> bool:
    dest = len(nums)-1
    i = 0
    jump = nums[i] 
    if jump >= dest: return True
    if jump == 0: return False
    while i <= dest:
        j = i+1
        max_j = nums[j] + j
        max_j_i = j
        while j <= dest and j <= jump+i:
            if nums[j] == 0:
                j+=1
                continue
            if nums[j] + j >= dest: return True
            if nums[j] + j >= max_j:
                max_j = nums[j] + j
                max_j_i = j
            j+=1
        i = max_j_i
        jump = nums[i]
        if jump == 0:return False
    return True

def jump(nums: list[int]) -> int:
    count_ler = 1
    dest = len(nums)-1
    i = 0
    jump = nums[i]
    if dest == 0: return 0
    if jump >= dest: return count_ler
    while i <= dest:
        j = i+1
        max_j = nums[j] + j
        max_j_i = j
        while j <= dest and j <= jump+i:
            if nums[j] == 0:
                j+=1
                continue
            if nums[j] + j >= dest: return count_ler + 1
            if nums[j] + j >= max_j:
                max_j = nums[j] + j
                max_j_i = j
            j+=1
        i = max_j_i
        jump = nums[i]
        count_ler+=1
        print(i, jump)
    return count_ler

def singleNumber(nums: list[int]) -> int:
    nums = sorted(nums)
    for i in range(0, len(nums)-1,2):
        if nums[i] != nums[i+1]:
            return nums[i]
    return nums[-1]

def hammingWeight(n: int) -> int:
    count = 0
    while n != 0:
        if n%2 == 1:
            count+=1
        n = n//2
    return count

def addBinary(a: str, b: str) -> str:
    add = ''
    i = len(a)-1
    j = len(b)-1
    carry = False
    while i >= 0 and j >= 0:
        add_bit = int(a[i]) + int(b[j])
        if carry:
            add_bit += 1
        if add_bit <= 1:
            add = str(add_bit) + add
            carry = False
        elif add_bit == 2:
            add = '0' + add
            carry = True
        else:
            add = '1' + add
            carry = True
        i-=1
        j-=1
    
    while i >= 0:
        add_bit = int(a[i])
        if carry:
            add_bit += 1
        if add_bit <= 1:
            add = str(add_bit) + add
            carry = False
        if add_bit == 2:
            add = '0' + add
            carry = True
        i-=1
    while j >= 0:
        add_bit = int(b[j])
        if carry:
            add_bit += 1
        if add_bit <= 1:
            add = str(add_bit) + add
            carry = False
        if add_bit == 2:
            add = '0' + add
            carry = True
        j-=1
    if carry:
        add = '1'+add
    return add

def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    if len(gas) == 1 and gas[0] - cost[0] >= 0 :
        return 0 
    i = 0
    while i < len(gas):
        if gas[i] - cost[i] <= 0:
            i+=1
            continue
        else:
            j = i
            fuel = 0
            fuel += gas[i]
            while (i-1)%len(gas) != j%len(gas):
                fuel -= cost[j%len(gas)]
                if fuel < 0:
                    break
                j+=1
                fuel += gas[j%len(gas)] 
            if fuel-cost[j%len(gas)] >= 0:
                return i
            i=j
    return -1

def candy(ratings: list[int]) -> int:
    if len(ratings) == 1: return 1
    candy = [1]*len(ratings)
    i = 0
    while i < len(ratings):
        if i == 0:
            if ratings[i] > ratings[i+1]:
                if candy[i] <= candy[i+1]:
                    candy[i] = candy[i+1] + 1
        elif i == len(ratings)-1:
            if ratings[i] > ratings[i-1]:
                if candy[i] <= candy[i-1]:
                    candy[i] = candy[i-1] + 1
        else:
            if ratings[i] > ratings[i+1]:
                if candy[i] <= candy[i+1]:
                    candy[i] = candy[i+1] + 1
            if ratings[i] > ratings[i-1]:
                if candy[i] <= candy[i-1]:
                    candy[i] = candy[i-1] + 1
        i+=1
    i = len(ratings)-1
    while i >= 0:
        if i == 0:
            if ratings[i] > ratings[i+1]:
                if candy[i] <= candy[i+1]:
                    candy[i] = candy[i+1] + 1
        elif i == len(ratings)-1:
            if ratings[i] > ratings[i-1]:
                if candy[i] <= candy[i-1]:
                    candy[i] = candy[i-1] + 1
        else:
            if ratings[i] > ratings[i+1]:
                if candy[i] <= candy[i+1]:
                    candy[i] = candy[i+1] + 1
            if ratings[i] > ratings[i-1]:
                if candy[i] <= candy[i-1]:
                    candy[i] = candy[i-1] + 1
        i-=1
    candy_count = 0
    for count in candy:
        candy_count+=count        
    return candy_count

# Q122 - M
def maxProfit_2(prices: list[int]) -> int:
    if len(prices) == 1: return 0
    total_p = 0
    i = 0
    while i < len(prices)-1:
        curren_p = prices[i+1]-prices[i]
        if curren_p > 0:
            total_p += curren_p
        i+=1
    return total_p

# Q13 - E
def romanToInt(s: str) -> int:
    value_convert = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000}
    n = 0
    i = 0
    while i < len(s)-1:
        cur, next = value_convert.get(s[i]), value_convert.get(s[i+1])
        if cur < next:
            n += next - cur
            i+=2
        else:
            n += cur
            i+=1
    if i < len(s):
        v = value_convert.get(s[i])
        n+=v
    return n

# Q12 - M
def intToRoman(num: int) -> str:
    value_convert = {
        1: 'I',
        5: 'V',
        10: 'X',
        50: 'L',
        100: 'C',
        500: 'D',
        1000: 'M'}
    s = ''
    digit_len = int(str(math.log10(num) + 1).split('.')[0])
    for i in range(1, digit_len):
        pass


def hIndex(citations: list[int]) -> int:
    # sum = 0
    # for citation in citations:
    #     sum+=citation
    # if sum < 2:
    #     return sum
    i = 0
    h = 0
    count_to_h = 0
    while h <= len(citations) and i < len(citations):
        if citations[i] >= h:
            count_to_h+=1
        if count_to_h >= h:
            i = -1
            h+=1
            count_to_h = 0
        if i == len(citations)-1:
            return h-1
        i+=1
    return h-1
    



c1 = [3,0,6,1,5]
c2 = [1,3,1]
c3 = [0,1]
c4 = [1]
c5 = [5,5,5,5,5]
print(hIndex(c1))
print(hIndex(c2))
print(hIndex(c3))
print(hIndex(c4))
print(hIndex(c5))