---
title: 'Guessing Game'
date: '2024-04-23T04:06:34+12:00'
# weight: 1
# aliases: ["/first"]
tags: ["python", "coding", "game"]
author: "simmeon"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Making a simple Python guessing game."
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Building a Number Guessing Game in Python

Let's make a simple game! Here's how:

```python {linenos=inline,hl_lines=[1,3]}
import random

secret_number = random.randint(1, 10)

while True:
    guess = int(input("Guess a number between 1 and 10: "))
    if guess == secret_number:
        print("You win!")
        break
    else:
        print("Nope, try again.")
```

The highlighted lines show how we can generate a random integer between 1 and 10.

We then enter a while loop to keep getting guesses from the player until they get it right. We convert the input into an integer and then check if it is the same as our secret number. If it it, the game is over and we break out of the loop. If it isn't, we ask for another guess.



