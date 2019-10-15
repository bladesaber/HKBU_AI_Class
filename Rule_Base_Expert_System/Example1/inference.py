import re
from Rule_Base_Expert_System.Example1.basic_rules import Rule
from Rule_Base_Expert_System.Example1.cv_handler import image_Handler, FactGenerator

re_a_rule = re.compile(r'(\{.*?\})')
re_if_part = re.compile(r'^\{IF:\s+(\[.*?\])')
re_then_part = re.compile(r'THEN:\s+(\'.*?\')')
re_description_part = re.compile(r'DESCRIPTION:\s+(\'.*?\')')

def generate_a_rule(rule):
    if_part_list = (re_if_part.findall(rule)[0]).split('\'')
    del (if_part_list[0])
    del (if_part_list[-1])
    if_part_temp = ''.join(if_part_list)
    if_part = if_part_temp.split(',')

    # if_part = ''
    # exec('if_part = ' + re_if_part.findall(rule)[0])

    then_part = re_then_part.findall(rule)[0]
    description_part = re_description_part.findall(rule)[0]

    return Rule(if_part, then_part.strip('"\''), description_part.strip('"\''))

def separate_rules(rules):
    rule_group = re_a_rule.findall(rules)
    return list(map(generate_a_rule, rule_group))

def read_rules(rule_file=None):
    with open('D:\HKBU_AI_Classs\Rule_Base_Expert_System\Example1/rules/rules.txt') as f:
        rules = ''.join(f.read().splitlines())
    return separate_rules(rules)

class Engine:
    '''backforward inference 技术'''
    def __init__(self, rule_library, fact_library):
        self.target = None
        self.rule_library = rule_library
        self.fact_library = fact_library
        self.condition_stack = []            # using Python list as stack
        self.matched_facts = {}
        self.hit_rules = {}

    # -------------------------------------------------------------
    # backforward inference 技术 使用从左往右的深度优先方法
    def push_condition_stack(self, condition):
        self.condition_stack.append(condition)

    def pop_condition_stack(self):
        return self.condition_stack.pop()

    def top_condition_stack(self):
        try:
            return self.condition_stack[-1]
        except IndexError:
            return None

    def condition_stack_is_empty(self):
        return not self.condition_stack

    # ---------------------------------------------------------------
    def __match_facts__(self, ant, contour_index):
        facts = self.fact_library['Contour' + str(contour_index)]

        # print('ant: ',ant.encode('utf-8'))
        # for fact in facts.line_facts:
        #     print('line: ',fact.fact.encode('utf-8'), ant==fact)
        # for fact in facts.angle_facts:
        #     print('angle: ',fact.fact.encode('utf-8'), ant==fact)

        matched = False
        for fact in facts.line_facts:
            if ant.replace(' ','') == fact.fact.replace(' ',''):
                matched = True
                self.matched_facts['Contour' + str(contour_index)].append(fact)
                return matched
        for fact in facts.angle_facts:
            if ant.replace(' ','') == fact.fact.replace(' ',''):
                matched = True
                self.matched_facts['Contour' + str(contour_index)].append(fact)
                return matched

        return matched

    # 触发 knowledge base 的哪个条件
    def __hit_consequent__(self, cons, contour_index):
        hit = False
        for rule in self.rule_library:
            '''
           1, 寻找这个结论在 knowledge database 的索引
           2，根据索引寻找这个结论的前提
           3，判断这个前提与个体的元知识是否匹配，不匹配则放入堆栈中
           4，如果最后前推到元知识，而元知识是不包含在 knowledge database 中的，会返回False
           '''
            if cons == getattr(rule, 'consequent'):
                hit = True
                self.hit_rules['Contour' + str(contour_index)].append(rule)
                ants = getattr(rule, 'antecedent')
                for ant in ants:
                    if not self.__match_facts__(ant, contour_index):
                        self.push_condition_stack(ant)
                break
        return hit

    def goal(self, target):
        self.target = target
        self.condition_stack = []
        self.matched_facts = {}
        self.hit_rules = {}
        self.push_condition_stack(target)

    def run(self, contour_index):
        self.matched_facts['Contour' + str(contour_index)] = []
        self.hit_rules['Contour' + str(contour_index)] = []
        found = True
        while self.condition_stack:
            # print(self.condition_stack)
            cons = self.pop_condition_stack()
            # 如果有一个命题产生 False 则 __hit_consequent__ 返回False 则命题不成立
            if not self.__hit_consequent__(cons, contour_index):
                found = False
                break
        self.condition_stack = [self.target]
        if found:
            return 'SUCCESS: Find Required Shape in Contour%d' % contour_index
        else:
            self.matched_facts['Contour' + str(contour_index)] = []
            self.hit_rules['Contour' + str(contour_index)] = []
            return 'ERROR: No Sufficient Facts in Contour%d' % contour_index

    def get_rule_library(self):
        return '\n'.join(list(map(str, self.rule_library)))

    def __str__(self):
        s = ''
        s += ('*' * 60 + '\n')
        s += 'Shape Detection Engine -- a Rule-based Expert System\n'
        s += '--------------- Below is all the rules ---------------\n'
        s += self.get_rule_library()
        s += '---------------- End of all the rules ----------------\n'
        return s

def setup_engine(image_source):
    handler = image_Handler(image_source)
    handler.generate_contour_dict()
    generator = FactGenerator(handler.contour_dict)
    generator.generate_fact()
    r = read_rules('rules.txt')
    e = Engine(r, generator.handled_facts)
    return e

def set_goal(e, my_goal):
    e.goal(my_goal)

def main_run(e):
    results = []
    for i in range(len(e.fact_library)):
        result = e.run(i)
        results.append(result)
    return results, e.matched_facts, e.hit_rules

if __name__ == '__main__':
    # test
    # e = setup_engine('D:\HKBU_AI_Classs\Rule_Base_Expert_System\Example1/test/test00.png')
    e = setup_engine('D:\HKBU_AI_Classs\Rule_Base_Expert_System\Example1/test/test01.png')
    set_goal(e, 'the shape is right triangle')
    results, matched_facts, hit_rules = main_run(e)
    for result in results:
        print('result: ',result)
    for fact in matched_facts['Contour0']:
        print('fact: ',fact.fact)
    for rule in hit_rules['Contour0']:
        print('rule: ',rule)
    # print('\n')
    # set_goal(e, 'the shape is triangle')
    # results, matched_facts, hit_rules = main_run(e)
    # for result in results:
    #     print(result)
    pass

# e = setup_engine('D:\HKBU_AI_Classs\Rule_Base_Expert_System\Example1/test/test01.png')
# facts = e.fact_library['Contour0']
# for fact in facts.line_facts:
#     print('line: ',fact.fact)
# for fact in facts.angle_facts:
#     print('angle: ',fact.fact)
