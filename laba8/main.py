from bs4 import BeautifulSoup as bs
import codecs

# открываем документ
doc = bs(codecs.open('web.html', encoding='utf-8', mode='r').read(), 'html.parser')
selectors = {
    'title': '.mailBlock_h h1',
    'other': '.random_hint',}


def get(doc, param):
    return doc.select(param)[0].decode_contents().strip()


title = get(doc, selectors['title'])
tags = list(map(lambda x: x.decode_contents().strip(), doc.select('div.widget ul li a')))
other = get(doc, selectors['other'])
news = list(map(lambda x: x.decode_contents().strip(), doc.select('ul.footer_nav li a')))


# вывод на экран
print('\nЗаголовок статьи:', title)
print('Теги:', tags)
print('\nКстати,', other)
print('\nЕщё:', news)
