---
title: "How to Add a 'Back to top' Footer to Your Hugo Posts"
date: 2022-08-07T12:38:11-07:00
draft: false
categories: ['Hugo',]
tags: ['Hugo', 'HTML', 'Web Development']
---
This post will detail how to add a footer that says "Back to Top" on all of the posts in your Hugo site. This method goes through directly altering html within your theme, which may or may not be advisable. I'm sure there are other methods. Anyways...
Within my Hugo site's root directory, I have a folder, `themes`, where [the theme I use](https://github.com/austingebauer/devise) lives. If your theme is structured similarly, you'll want to go from `themes` into your actual theme's folder, in my case `devise`, then into `layouts`.
```bash {hl_lines=[7]}
themes
└───devise
    ├───archetypes
    ├───assets
    ├───exampleSite
    ├───images
    ├───layouts
    │   ├───categories
    │   ├───partials
    │   │   └───helpers
    │   ├───post
    │   ├───tag
    │   └───_default
    ├───node_modules
    │   └───bootstrap
    └───static
```
In the `layouts` folder, where you'll see a number of sub-folders. Navigate to partials, within which you should see a file, `footer.html` that contains the html that defines the footer for every page on the site.
```bash {hl_lines=[10]}
layouts
│   404.html
│   index.html
│
├───categories
│       list.html
│
├───partials
│   │   category-posts.html
│   │   footer.html
│   │   head.html
│   │   header.html
│   │
│   └───helpers
│           katex.html
│
├───post
│       list.html
│       single.html
│
├───tag
│       list.html
│
└───_default
        baseof.html
        list.html
        single.html
```

In my case, this file just contains html code for a footer. The code -- lines 2-5 -- are highlighted. Since this footer appears on every page and I only want the "Go back to top" line to appear on my actual blog posts, I have to add logic that determines whether the footer is being rendered on a blog post. You can accomplish using a [Page Variable in Hugo](https://gohugo.io/variables/page/):
>Page-level variables are defined in a content file’s front matter, derived from the content’s file location, or extracted from the content body itself.

The key here is "derived from the content's file location. You can use the variable `.IsPage` to check whether you're on a blog post page vs. your home or index page.


```html {linenos=inline, hl_lines=["2-5"]}
<footer class="text-center pb-1">
    {{ if .IsPage }}
    <p></p>
    <p><a href="#content">&uarr;Back to Top&uarr;</a></p>
    {{ end }}
    <small class="text-muted">
        {{ if .Site.Copyright }}
            {{ .Site.Copyright | safeHTML }}
        {{ else }}
            {{ "&copy; Copyright Year, Your Name" | safeHTML }}
        {{ end }}
        <br>
        Powered by <a href="https://gohugo.io/" target="_blank">Hugo</a>
        and <a href="https://github.com/austingebauer/devise" target="_blank">Devise</a>
    </small>
</footer>
```
With the logic set up (lines 2 and 5), I can put in the actual html I want, in this case:
```html
<p></p>
<p><a href="#content">&uarr;Back to Top&uarr;</a></p>
```
some dead space and the "Back to Top" text wrapped in up arrows. You can see that rendered at the bottom of this page.