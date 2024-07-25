class Sdoc:
    '''
    strictdoc template class
    '''
    def __init__(self,title,version,classification):
        self.header=f'''[DOCUMENT]
TITLE: {title}
VERSION: {version}
CLASSIFICATION: {classification}

'''
        self.reqs=[]
    def new_req(self,uid,title,statement,parent=None):
        link = f'''
RELATIONS:
- TYPE: Parent
  VALUE: {parent}''' if parent is not None else ''
        self.reqs.append(f'''[REQUIREMENT]
UID: {uid}
TITLE: {title}
STATEMENT: >>>
{statement}
<<<{link}
''')
    def write(self,file_path):
        with open(file_path,'w') as f:
             f.write(str(self))

    def __str__(self):
        return self.header+'\n'.join(self.reqs)
