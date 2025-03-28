import math
from time import sleep
from typing import List

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
        while(j < len(prices) and prices[j] - prices[i] > 0):
            if(prices[j] - prices[i] > max_profit):
                max_profit = prices[j] - prices[i]
            j+=1
        i=j
        j+=1
    return max_profit

def binary_search(nums, t, offset=0):
    if(nums[len(nums)//2] == t or len(nums) == 1):
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
    value_list = [1,5,10,50,100,500,1000]
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
    for i in range(1, digit_len+1):
        i_digit = num % (10**i)
        if i_digit == 0:
            continue
        f = value_convert.get(i_digit)
        if f:
            s = f + s
        # if i_digit is - 4,9,40,90,400,900
        elif value_convert.get(i_digit + 10**(i-1)):
            sub_1 = value_convert.get(10**(i-1))
            sub_2 = value_convert.get(i_digit + 10**(i-1))
            s = (sub_1+sub_2) + s
        else:
            tmp = i_digit
            while tmp != 10**(i-1) and not value_convert.get(tmp) :
                part_num = value_convert.get(10**(i-1))
                s = part_num + s
                tmp -= 10**(i-1)
            s = value_convert.get(tmp) + s
        num-=i_digit
    return s

# Q274 - M
def hIndex(citations: list[int]) -> int:
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
    
# Q125 - E
def isPalindrome(s: str) -> bool:
    s = s.lower()
    i = 0
    j = len(s)-1
    while i<j:
        if s[i].isalnum() and s[j].isalnum():
            if s[i] != s[j]:
                return False
            i+=1
            j-=1
        elif not s[i].isalnum():
            i+=1
        else:
            j-=1
    return True

# Q11 - M
def maxArea(height: list[int]) -> int:
    i = 0
    j = len(height)-1
    max_area = j * min(height[i], height[j])
    while i < j:
        if height[i] < height[j]:
            i+=1
        elif height[i] > height[j]:
            j-=1
        else:
            i+=1
            j-=1
        if (j-i) * min(height[i], height[j]) > max_area:
            max_area = (j-i) * min(height[i], height[j])
    return max_area

# Q15 - M
def threeSum(nums: list[int]) -> list[list[int]]:
    ans = []
    nums = sorted(nums)
    i = 0
    print(nums)
    while i < len(nums)-2:
        l = i+1
        r = len(nums)-1
        first = nums[i]
        while l < r:
            if first + nums[r] + nums[l] == 0:
                ans.append([first,nums[r],nums[l]])
                l+=1
                while l < r-1 and nums[l] == nums[l+1]:
                    l+=1
            elif first + nums[r] + nums[l] > 0:
                r-=1
            else:
                l+=1
        i+=1
        while i < len(nums)-2 and nums[i-1] == nums[i]:
            i+=1
    return ans

# Q42 - H
def trap(height: list[int]) -> int:
    max_water = 0
    i = 0 
    while i < len(height)-1:
        if height[i] > 0:
            pass    
        else:
            i+=1
    return max_water
h1 = [0,1,0,2,1,0,1,3,2,1,2,1]
h2 = [4,2,3]
# print(trap(h1))
# print(trap(h2))    

# Q1 - E
def twoSum(nums: list[int], target: int) -> list[int]:
    two_for_t = {}
    for i,n in enumerate(nums):
        two_for_t[n] = i
    for i,n in enumerate(nums):
        if two_for_t.get(target-n) != None:
            sec = two_for_t.get(target-n)
            if sec != i:
                return [i, sec]

# Q202 - E
def isHappy(n: int) -> bool:
    mem_cycle = {}
    cycle= False
    while not cycle:
        square_sum = 0
        for digit in str(n):
            square_sum+=int(digit)**2
        if square_sum == 1:
            return True
        found = mem_cycle.get(square_sum)
        if found != None:
            return False
        mem_cycle[square_sum] = 1
        n = square_sum

# Q242 - E
def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):return False
    letters = {}
    for i in range(len(s)):
        count = letters.get(s[i],0)
        letters[s[i]] = count+1
        count = letters.get(t[i],0)
        letters[t[i]] = count-1
    for v in letters.values():
        if v:
            return False
    return True

# Q219 - E
def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    dup = {}
    for i, n in enumerate(nums):
        exist = dup.get(n)
        if exist is None:
            dup[n] = i
        else:
            nearby = abs(i - exist)
            if nearby <= k:
                return True
            dup[n] = i
    return False

# Q290 - E
# find if there is bijection functon from pattern to s
def wordPattern(pattern: str, s: str) -> bool:
    src = {}
    dst = {}
    s = s.split(' ')
    if len(s) != len(pattern): return False
    for i, p in enumerate(pattern):
        exist_s = src.get(p)
        if exist_s is None:
            src[p] = s[i]
        elif exist_s != s[i]: # its not function - 1 goes to 2 diffrent item in dst 
            return False
        exist_d = dst.get(s[i])
        if exist_d is None: # its first map on p in pattern
            dst[s[i]] = p
        elif p != exist_d: # its not bijection - 2 diffrent items goes to 1 item
            return False
    return True

# Q128 - M
def longestConsecutive(nums: list[int]) -> int:
    max_consecutive = 0
    consecutive = set()
    for n in nums:
        consecutive.add(n)
    while consecutive:
        item = consecutive.pop()
        up = False
        down = False
        if len({item+1}.intersection(consecutive)) > 0:
            up = True
        if len({item-1}.intersection(consecutive)) > 0:
            down = True
        counter = 0
        offset_i = 1
        while up:
            if len({item+offset_i}.intersection(consecutive)) > 0:
                consecutive.remove(item+offset_i)
                offset_i+=1
                counter+=1
            else:
                up = False  
        offset_i = 1
        while down:
            if len({item-offset_i}.intersection(consecutive)) > 0:
                consecutive.remove(item-offset_i)
                offset_i+=1
                counter+=1
            else:
                down = False
        if counter+1 > max_consecutive:
            max_consecutive = counter+1
    return max_consecutive
    
# Q49 - M
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    group = {}
    i = 0
    while i < len(strs):
        s = "".join(sorted(strs[i]))
        exist  = group.get(s)
        if exist is None:
            group[s] = [strs[i]]
        else:
            group[s].append(strs[i])
        i+=1
    return list(group.values())
        
def minSubArrayLen(target: int, nums: list[int]) -> int:
    i = 1
    while i <= len(nums):
        j = 0
        sum = 0
        while j+i <= len(nums):
            for n in nums[j:j+i]:
                sum += n
            if sum >= target:
                return i
            j+=1
            sum = 0
        i+=1
    return 0

def summaryRanges(nums: list[int]) -> list[str]:
    ranges = []
    i = 0
    while i < len(nums):
        start = nums[i]
        while i < len(nums)-1 and nums[i] - nums[i+1] == -1:
            i+=1
        if start == nums[i]:
            ranges.append(f'{start}')
        else:
            ranges.append(f'{start}->{nums[i]}')
        i+=1
    
    return ranges

# Q207 - M
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    course_g = {i:[] for i in range(numCourses)}
    cycle_c = set()
    for pre in prerequisites:
        if pre[0] in course_g:
            course_g[pre[0]].append(pre[1])
        else:
            course_g[pre[0]] = [pre[1]]
    def dfs_course(course):
        if course in cycle_c:
            return False
        if not course_g[course]:
            return True
        cycle_c.add(course)
        for c in course_g[course]:
            if not dfs_course(c):
                return False
        cycle_c.remove(course)
        course_g[course] = []
        return True

    for i in range(numCourses):
        if not dfs_course(i):
            return False
    return True

# Q198 - M
def rob(nums: list[int]) -> int:
    opt1 = 0
    opt2 = 0
    for i in range(len(nums)):
        tmp = max(opt1 + nums[i], opt2)
        opt1 = opt2
        opt2 = tmp
    return opt2

# Q139 - M
def wordBreak(s: str, wordDict: list[str]) -> bool:
    dp = [False] * (len(s)+1)
    dp[-1]=True
    i = len(s)-1
    while i >= 0:
        for w in wordDict:
            if (i+len(w)) <= len(s) and s[i:i+len(w)] == w:
                dp[i] = dp[i+len(w)]
            if dp[i]:
                break
        i-=1
    return dp[0]


# Q5 - M
def longestPalindrome(s: str) -> str:
    res = ''
    for i in range(len(s)):
        l = i
        r = i
        while l >= 0 and r < len(s) and s[r] == s[l]:
            l-=1
            r+=1
        if len(s[l+1:r]) > res:
            res = s[l+1:r]
        l = i
        r = i+1
        while l >= 0 and r < len(s) and s[r] == s[l]:
            l-=1
            r+=1
        if len(s[l+1:r]) > res:
            res = s[l+1:r]
    return res

# Q647 - M
def countSubstrings(s: str) -> int:
    counter = 0
    for i in range(len(s)):
        l = i
        r = i
        while l >= 0 and r < len(s) and s[r] == s[l]:
            l-=1
            r+=1
            counter+=1 
        l = i
        r = i+1
        while l >= 0 and r < len(s) and s[r] == s[l]:
            l-=1
            r+=1
            counter+=1
    return counter

# Q322 - M
def coinChange(coins: list[int], amount: int) -> int:
    dp = [amount + 1] * (amount+1)
    dp[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if i - c >= 0:
                dp[i] = min(dp[i], 1 + dp[i-c])
    if dp[-1] > amount:
        return -1
    return dp[-1]

def longestCommonSubsequence(text1: str, text2: str) -> int:
    counter1 = {}
    counter2 = {}
    common = 0
    for c in text1:
        if c in counter1:
            counter1[c] += 1
        else:
            counter1[c] = 1
    for c in text2:
        if c in counter2:
            counter2[c] += 1
        else:
            counter2[c] = 1
    if len(text1) < len(text2):
        for k in counter1.keys():
            if k in counter2:
                common += min(counter1[k], counter2[k])
    else:
        for k in counter2.keys():
            if k in counter1:
                common += min(counter1[k], counter2[k])
    return common    
    
# Q57 - M
def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    if intervals == []:
        return [newInterval]
    if newInterval == []:
        return intervals
    i = 0
    while i < len(intervals):
        if intervals[i][1] < newInterval[0]:
            i += 1
        elif intervals[i][0] > newInterval[1]:
            return intervals[0:i] + [newInterval] + intervals[i:]
        elif intervals[i][0] <= newInterval[0] and intervals[i][1] >= newInterval[1]:
            return intervals
        else:
            j = i
            while j < len(intervals) and intervals[j][0] <= newInterval[1]:
                j+=1
            # print('******')
            return intervals[0:i] + [[min(intervals[i][0], newInterval[0]),max(newInterval[1], intervals[j-1][1])]]+ intervals[j:]
    return intervals + [newInterval]

a1 = [[1,3],[6,9]]
b1 =[2,5]
a2 = [[1,2],[3,5],[6,7],[8,10],[12,16]]
b2 = [4,8]
a3 = [[1,5]]
b3 = [6,8]
b4 = [0,3]
a4 = [[3,5],[12,15]]
b4 = [6,6]
print(insert(a1,b1))
print(insert(a2,b2))
print(insert(a3,b3))
print(insert(a3,b4))
print(insert(a4,b4))