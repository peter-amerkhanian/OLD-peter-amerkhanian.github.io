---
title: "How to Add a 'Back to top' Footer to Your Hugo Posts"
date: 2022-08-07T12:38:11-07:00
draft: false
categories: ['Hugo Development',]
tags: ['Hugo', 'HTML', 'Web Development']
---
This post will detail how to add a footer that says "Back to Top" on all of the posts in your Hugo site. This method involves taking advantage of Hugo's "partials" feature, which allows you to add customizations to your Hugo page without directly altering your theme's html. Partials are utilized in a hierarchical system, where Hugo sees if you have a custom file written for a given partial (e.g., your `footer.html`), and if not,renders the partial included in your theme. [Hugo docs](https://gohugo.io/templates/partials/) explain the partials hierarchy as follows:
>Partial templates—like single page templates and list page templates—have a specific lookup order. However, partials are simpler in that Hugo will only check in two places:
>- `layouts/partials/*<PARTIALNAME>.html`
>- `themes/<THEME>/layouts/partials/*<PARTIALNAME>.html`  
>
>This allows a theme’s end user to copy a partial’s contents into a file of the same name for further customization.

To start,   

Within my Hugo site's root directory, I have a folder, `themes`, where [the theme I use](https://github.com/austingebauer/devise) lives.
```bash {hl_lines=[9]}
root
├───.github
├───archetypes
├───content
├───layouts
├───public
├───resources
├───static
└───themes
```

If your theme is structured similarly, you'll want to go from `themes` into your actual theme's folder, in my case `devise`, then into `layouts`.
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
In the `layouts` folder, where you'll see a number of sub-folders. Navigate to partials. From [the Hugo docs](https://gohugo.io/templates/partials/):
>Partials are smaller, context-aware components in your list and page templates that can be used economically to keep your templating DRY.

Within your partials folder, you should see a file, `footer.html` that contains the html that defines the footer for every page on the site.
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

In my case, this file just contains html code for a footer. If that exists **Copy that file, `footer.html`**. From there, you'll want to get back to your project's root, then navigate into your `layouts` folder. This folder functions essentially the same as the `layouts` folder that exists in your theme, but whatever is in this one takes priority when Hugo renders your site. Create a folder within `layouts` called `partials` if it doesn't already exist. Within `partials`, **paste the `footer.html` file you previously copied**


```bash {hl_lines=[9]}
ROOT_OF_PROJECT
│ 
├───.github
│ 
├───archetypes
├───content
│ 
├───layouts
│   ├───partials
│   └───shortcodes
├───public
│ 
├───resources
│ 
├───static
│ 
└───themes
```


Now that we've copied the `footer.html` file into our principal `layouts/partials` directory, let's actually edit it. You can see the full code below. Since this footer appears on every page and I only want the "Go back to top" line to appear on my actual blog posts, I have to add logic that determines whether the footer is being rendered on a blog post. You can accomplish this using a [Page Variable in Hugo](https://gohugo.io/variables/page/):
>Page-level variables are defined in a content file’s front matter, derived from the content’s file location, or extracted from the content body itself.

The key here is "derived from the content's file location." You can use the variable `.IsPage` to check whether you're on a blog post page vs. your home or index page. The code for executing that logic is highlighted below:


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
The footer for blog posts just contains some dead space and the "Back to Top" text wrapped in up arrows. At this point you can save and test it out on your local development server. You can see what mine looks like fully rendered at the bottom of this page.