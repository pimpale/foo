from commands.BaseCommand import BaseCommand, CallbackType
import subprocess
import time
import os

class BrowseCommand(BaseCommand):
    def __init__(self):
        super().__init__(
            xml_tag="browse",
            description="""You have the ability to browse the internet using a headless CLI browser. Use this command to learn about how to browse. Example: <browse></browse>""",
            callback=None,
        )

        subprocess.Popen(
            ["nohup", "browse-start"],
            stdout=open("/dev/null", "w"),
            stderr=open("/dev/null", "a"),
            preexec_fn=os.setpgrp,
        )
        time.sleep(0.2)

    def _run(self, content: str) -> str:
        if content != "":
            return "ERROR: The browse command does not take any arguments."

        return """To browse the web, use the `browse-*` family of commands wrapped in the <bash></bash> XML tags.

Documentation for each command in the family:

To open a URL in the browser:
```
Usage: browse-goto [OPTIONS] URL

  Goes to the url URL

Options:
  --help  Show this message and exit.
```

To click an element:
```
Usage: browse-click [OPTIONS] ID

  Clicks on the element ID

Options:
  --help  Show this message and exit.
```

To type in text:
```
Usage: browse-type [OPTIONS] ID TEXT

  Types the text TEXT in the element ID. Surround TEXT in quotes.

Options:
  --enter  Press enter after typing.
  --help   Show this message and exit.
```

To scroll up or down in the page:
```
Usage: browse-scroll [OPTIONS] {up|down}

  Scrolls the page in the DIRECTION direction

Options:
  --help  Show this message and exit.
```

To navigate in browser history:
```
Usage: browse-navigate [OPTIONS] {back|forward}

  Navigates browser history in the DIRECTION direction

Options:
  --help  Show this message and exit.
```

To view the page contents again:
```
Usage: browse-observe [OPTIONS]

  Observes the page

Options:
  --help  Show this message and exit.
```

To reload the page:
```
Usage: browse-reload [OPTIONS]

  Reloads the page

Options:
  --help  Show this message and exit.
```

Demonstration:

**ASSISTANT**:
<bash>browse-goto "https://en.m.wikipedia.org"</bash>

**USER**:
BASH OUTPUT:
Viewing URL: https://en.m.wikipedia.org/wiki/Main_Page

You are only viewing part of the page. Scroll percentage: 0.00%

Page Content:


(0) button: ''hasPopup: menu expanded: False
[Wikipedia](1) 
(2) searchbox: 'Search Wikipedia'keyshortcuts: Alt+f
(3) main: ''
        [Banner logo You are invited to join the Bay Area Wiki-Picnic at Mission Dolores Park on Saturday, August 17!](4) 
        (5) generic: 'Hide'
        (6) heading: 'Welcome to Wikipedia'
        , the [free](9) [encyclopedia](11) that [anyone can edit](13) . [6,866,238](15) articles in [English](17)
        (18) heading: 'From today's featured article'
        [T3, a sister ship of T2](19) T3 , a sister ship of T2 [T2](23) was a [torpedo boat](25) of the [Royal Yugoslav Navy](27) . Originally a [250t-class torpedo boat](29) of the [Austro-Hungarian Navy](31) , commissioned on 11 August 1914 as 77T , she saw active service during World War I, performing [convoy](35) , patrol, escort, [minesweeping](37) and [minelaying](39) tasks, [anti-submarine operations](41) , and [shore bombardment](43) missions. Present in the [Bocche di Cattaro](45) during [the short-lived mutiny](47) by Austro-Hungarian sailors in early February 1918, members of her crew raised the [red flag](49) but took no other mutinous actions. The boat was part of the escort force for the Austro-Hungarian [dreadnought](51) [Szent István](53) when that ship was sunk by [Italian](55) torpedo boats in June 1918. Following [Austria-Hungary](57) 's defeat in 1918, the boat was allocated to the Navy of the [Kingdom of Serbs, Croats and Slovenes](59) , which became the Royal Yugoslav Navy in 1921, and was renamed T2 . During the [interwar period](63) , Yugoslav naval activity was limited by reduced budgets. Worn out after twenty-five years of service, T2 was [scrapped](67) in 1939. 
        (69) heading: 'In the news'
        (70) figure: ''
                [ATR 72-500 Voepass in August 2023](71) The [ATR 72](73) involved in the crash
        (75) ListMarker: '• '
        [Voepass Linhas Aéreas Flight 2283](76)  (aircraft pictured) crashes in the Brazilian [state of São Paulo](80) , killing all 62 people on board.
        (82) ListMarker: '• '
        [Sheikh Hasina](83)  resigns as the [prime minister of Bangladesh](85) following [anti-government protests](87) , and [Muhammad Yunus](89) is appointed leader of [an interim government](91) .
        (93) ListMarker: '• '
        Following  [a mass stabbing](95) in [Southport](97) , [far-right protesters riot](99) in England and Northern Ireland.
        (101) ListMarker: '• '
        The United States, Russia, and their respective allies agree to  [a prisoner exchange](103) of 26 people. [Ongoing](105) : [Israel–Hamas war](107) [timeline](109) [Russian invasion of Ukraine](110) [timeline](112) [Sudanese civil war](113) [timeline](115) [Summer Olympics](116)

**ASSISTANT**:

<bash>browse-type 2 "London" --enter</bash>

**USER**:
BASH OUTPUT:
Viewing URL: https://en.m.wikipedia.org/wiki/London

You are only viewing part of the page. Scroll percentage: 0.00%

Page Content:


(0) button: ''hasPopup: menu expanded: False
[Wikipedia](1) 
(2) searchbox: 'Search Wikipedia'keyshortcuts: Alt+f
(3) main: ''
        [Banner logo You are invited to join the Bay Area Wiki-Picnic at Mission Dolores Park on Saturday, August 17!](4) 
        (5) generic: 'Hide'
        (6) heading: 'London'
        [Article](7) [Talk](8)
        (9) button: 'Language'
                Language 
        (11) button: 'Watch'
                Watch 
        (13) button: 'View source'
                View source 
        (15) note: ''
                This article is about the capital city of England and the United Kingdom. For other uses, see  [London (disambiguation)](17) .
        London  ( [/ˈlʌndən/](21) [LUN-dən](23) ) [[6]](25) is the [capital](27) and [largest city](29) [[c]](30) of both [England](32) and the [United Kingdom](34) , with a population of 8,866,180 in 2022. [[2]](36) The [wider metropolitan area](38) is the largest in [Western Europe](40) , with a population of 14.9 million. [[7]](42) London stands on the [River Thames](44) in southeast England, at the head of a 50-mile (80 km) [estuary](46) down to the [North Sea](48) , and has been a major settlement for nearly 2,000 years. [[8]](50) Its ancient core and [financial centre](52) , the [City of London](54) , was founded by the [Romans](56) as [Londinium](58) and has retained its medieval boundaries. [[d]](60) [[9]](61) The [City of Westminster](63) , to the west of the City of London, has been the centuries-long host of the national [government](65) and [parliament](67) . London grew rapidly [in the 19th century](69) , becoming the world's [largest city at the time](71) . Since the 19th century, [[10]](73) [[11]](74) the name "London" has referred to the [metropolis](76) around the City of London, historically split between the [counties](78) of [Middlesex](80) , [Essex](82) , [Surrey](84) , [Kent](86) , and [Hertfordshire](88) , [[12]](90) which since 1965 has largely comprised the administrative area of [Greater London](92) , governed by [33 local authorities](94) and the [Greater London Authority](96) . [[e]](98) [[13]](99)
        (100) table: ''
                (101) row: ''
                        (102) columnheader: 'London'required: False
                                London 
                (104) row: ''
                        (105) gridcell: 'Capital city'required: False
                                [Capital city](106) 
                (107) row: ''
                        (108) gridcell: 'River Thames and Tower Bridge with The Shard and Southwark (left), and Tower of London and City of London (right) London Eye Nelson's Column St Paul's Piccadilly Circus Canary Wharf Palace of Westminster with Big Ben (right)'required: False
                                [](109) 
        As one of the world's major  [global cities](111) , [[14]](113) [[15]](114) London exerts a strong influence on world [art](116) , entertainment, [fashion](118) , commerce, finance, [education](120) , [healthcare](122) , [media](124) , science, technology, [tourism](126) , [transport](128) , and communications. [[16]](130) [[17]](131) Despite a post- [Brexit](133) exodus of stock listings from the [London Stock Exchange](135) , [[18]](137) London remains Europe's most economically powerful city [[19]](139) and [one of the world's major financial centres](141) . It hosts Europe's largest concentration of [higher education institutions](143) , [[20]](145) some of which are the highest-ranked academic institutions in the world: [Imperial College London](147) in [natural](149) and 
             
"""
