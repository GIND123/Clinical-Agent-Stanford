-- pandoc filter for the typeset manuscript: vector figures at full width, "Fig. N |" captions kept as written.
function Image(img)
  img.src = img.src:gsub("%.png$", ".pdf")
  img.attributes.width = "\\textwidth"
  return img
end
-- the Markdown links to repository files make no sense in print: keep their text
function Link(el)
  if el.target:match("^%.%./") or el.target:match("^docs/") or el.target:match("%.md$") or el.target:match("%.py$") then
    return el.content
  end
  return el
end
