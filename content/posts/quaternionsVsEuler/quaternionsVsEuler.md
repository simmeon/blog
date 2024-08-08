---
title: 'But Why Rotate With Quaternions?'
date: '2024-08-07'
# weight: 1
# aliases: ["/first"]
tags: ["quaternions", "euler", "orientation", "rotation", "interactive"]
author: "simmeon"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: false
description: "An interactive exploration of Euler angles, quaternions, and what rotation looks like."
canonicalURL: "https://simmeon.github.io/blog/post/quaternionsVsEuler.md"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: false
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/simmeon/blog/tree/main/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

<!-- Import three.js -->
<script type="importmap">
        {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.167.1/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.167.1/examples/jsm/"
            }
        }
</script>

<!-- Import KaTeX -->
{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

*This post is designed to be a follow on from the great [3Blue1Brown videos](https://www.youtube.com/watch?v=d4EgbgTm0Bg) and [interactive site](https://eater.net/quaternions) by Ben Eater about quaternions.* 

*If you don't want to go through all of that, the second part, [Quaternions and 3d rotation, explained interactively [5m58s]](https://www.youtube.com/watch?v=zjMuIxRvygQ), is most relevant to introducing the ideas we will be exploring.*

Testing three js shortcode

{{< threejs src="../code/test.js" canvasId="canvas1" height="400px" width="100%" >}}

End of shortcode 1

{{< threejs src="../code/test2.js" canvasId="canvas2" height="300px" width="80%" >}}

## Intuitive Rotation (Euler Angles)


## Problems With Euler Angles

### Gimbal Lock

### Order Ambiguity


## Quaternions

### What is a Quaternion?

### Why do Quaternions Rotate Things?

### Performing Quaternion Rotation