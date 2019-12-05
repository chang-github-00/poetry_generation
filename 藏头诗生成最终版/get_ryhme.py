class Ryhme(object):
    rhyme = dict()
    pitch = dict()
    rhy = -1

    def __init__(self, config):
        # print('now,start!geting rhyme and pitch')
        puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']
        nums = ['一',  '二',  '三', '四', '五', '六', '七', '八', '九', '十']

        rhyme_file = config.ryhme_file

        nlines = list()
        nnlines = list()

        with open(rhyme_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # print(len(lines))
            for n in range(0, len(lines)):
                line = lines[n]
                line = line.replace('　', '')
                line = line.replace('\n', '')
                line = line.replace(' ', '')
                # print(line, sep='')
                if len(line) <= 1:
                    continue
                nnlines.append(line)

        bj = 0
        for line in nnlines:
            tmp = ''
            for char in line:
                if char == '(' or char == '（':
                    bj = 1
                if bj == 0:
                    tmp += char
                if char == ')' or char == '）':
                    bj = 0
            nlines.append(tmp)

        cnt = 0
        p = 0
        # ping 0 ze 1
        for line in nlines:
            if line[0] in nums:
                cnt += 1
                continue
            if line[1] == '平':
                p = 0
                continue
            if line[1] == '声':
                p = 1
                continue
            if line[0] == '派':
                continue
            for char in line:
                self.rhyme[char] = cnt
                self.pitch[char] = p

        print('rhyme and putch got')
        '''
        ch = '爸'
        print(rhyme[ch],pitch[ch])
        '''
