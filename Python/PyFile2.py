# -*- coding:utf-8 -*-
import shelve
# 存数据
dbase = shelve.open("mydbase")
object1 = ['The', 'bright', ('side', 'of'), ['life']]
object2 = {'name': 'Brian', 'age': 33, 'motto': object1}
dbase['brian'] = object2
dbase['knight'] = {'name': 'Knight', 'motto': 'Ni!'}
dbase.close()
# 取数据
dbase = shelve.open("mydbase")
print(dbase.keys())
print(dbase['knight'])
dbase.close()