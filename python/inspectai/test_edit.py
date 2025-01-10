#%%
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import xml

@dataclass
class CommandOutput:
    output: str


def parse(content) -> CommandOutput:
        # problem: we are presented with a set of xml tags  (whatever is contained within <edit> </edit>)
        # We need to split the content into the individual tags
        # For each tag, we encode into a dictionary
        # finaly, we pass it into the edit_tool to execute the command
    
        d: defaultdict[str, Any] = defaultdict(lambda: None)
        element_stack: list[str] = []
    
        def end_element(name: str):
            element_stack.pop()
    
        def start_element(name:str, attrs):
            element_stack.append(name)
        
        def char_data(data):
            data = data.replace("&lt;", "<").replace("&gt;", ">")
            if d[element_stack[-1]] is None:
                d[element_stack[-1]] = data
            else:
                d[element_stack[-1]] += data
        
        p = xml.parsers.expat.ParserCreate()
        p.StartElementHandler = start_element
        p.EndElementHandler = end_element
        p.CharacterDataHandler = char_data
        
        try:
            p.Parse(content, True)
        except Exception as e:
            return CommandOutput(str(e))

        # parse view range (if it exists)
        if d["view_range"]:
            nums = [i for i in d["view_range"].strip("[]").split(",")]                
            if nums == [""]:
                d["view_range"] = []
            else:
                for n in nums:
                    if not n.strip().isdigit():
                        return CommandOutput("ERROR: view_range must contain only positive integers")
                d["view_range"] = [int(i) for i in nums]

        # parse insert line (if it exists)
        if d["insert_line"]:
            d["insert_line"] = int(d["insert_line"])
            
        for key in d:
            print('=================')
            print(key, d[key])
        

#%%

content = """
<edit>a
<cmd>string_replace</cmd>
<path>/testbed/tests/annotations/models.py</path>
<old_str> .filter(
pages__gt=400,
)
.annotate(book_annotate=Value(1))
.alias(book_alias=Value(1))
</old_str>
<new_str> .filter(
pages__gt=400,
)
.annotate(book_annotate=Value(1))
.values()
.alias(book_alias=Value(1))
</new_str>
</edit>
"""

parse(content)